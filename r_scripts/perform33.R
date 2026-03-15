#################################################################
# PERFORMANCE EVALUATION OF BINARY PREDICTIVE MODELS (USER-FRIENDLY)
#################################################################

# ------------ 1) GEREKLİ PAKETLERI YÜKLEME ------------ #

# install.packages("pacman")  # Eğer yüklü değilse
library(pacman)

pacman::p_load(
  dplyr, ggplot2, pROC, PRROC, yardstick, 
  rms, CalibrationCurves, ResourceSelection, rmda, 
  ROCR
)

# ------------ 2) KULLANICI İÇİN KOLAY AYARLAR ------------ #

# Lütfen kendi veri setinizi (mydata) yükleyin veya tanımlayın.
# Örnek: mydata <- read.csv("mydata.csv")

# Bu iki değişkeni kendi veri setinize göre ayarlayın:
outcome_col <- "y"   # Outcome değişkeni, 0/1 formatında
pred_col    <- "p"   # Model tahmini (olasılık)
cut         <- 0.40  # Karar eşiği
costratio   <- 6     # Yanlış negatif / yanlış pozitif maliyet oranı

# (Örneğin costratio=9 ve cut=0.1 tutarlı, çünkü cut=0.1 => odds=0.1/0.9=1/9,
#  costratio ~ 9 => Net Benefit formülünde tutarlı bir eşik.)

# ------------ 3) ÖRNEK: REKALİBRE ETMEK İSTERSENİZ (opsiyonel) ------------ #
# Lojistik regresyonla basit rekalibrasyon (logistic calibration):
# 'pmalwou' adında yeni bir kolon oluşturup kalibre edilmiş tahminleri içerecek.
# Örnek: mydata$pmalwou <- predict(lrm(y ~ qlogis(p)), data = mydata, type="fitted")

# ------------ 4) PERFORMANS FONKSİYONLARI ------------ #
# Aşağıdaki fonksiyonlar, orijinal kodların sadeleştirilmiş hâlidir.

# 4.1) Hızlı AUROC hesabı
fastAUC <- function(p, y) {
  x1 <- p[y == 1]
  x2 <- p[y == 0]
  r  <- rank(c(x1, x2))
  auc <- (sum(r[1:length(x1)]) - length(x1)*(length(x1)+1)/2) / 
    (length(x1)*length(x2))
  return(auc)
}

# 4.2) Average Precision (AP) hesabı
avgprec <- function(p, y) {
  probsort <- sort(p)
  prcpts   <- data.frame(matrix(NA, nrow = length(probsort), ncol = 3))
  
  for (i in seq_along(probsort)) {
    prcpts[i,1] <- sum(p[y == 1] >= probsort[i]) / sum(y == 1)       # recall
    prcpts[i,2] <- sum(p[y == 1] >= probsort[i]) / sum(p >= probsort[i]) # precision
  }
  for (i in seq_along(probsort)) {
    if (i == length(probsort)) {
      prcpts[i,3] <- prcpts[i,2] * prcpts[i,1]
    } else {
      prcpts[i,3] <- prcpts[i,2] * (prcpts[i,1] - prcpts[i+1,1])
    }
  }
  return(sum(prcpts[,3]))
}

# 4.3) Discrimination Metrics (AUROC, AUPRC, AP, pAUROC)
DiscPerfBin <- function(y, p, paucf = "se", paucr = c(1, 0.8)){
  cstat <- fastAUC(p, y)
  auprc <- pr.curve(scores.class0 = p[y==1], scores.class1 = p[y==0],
                    curve = TRUE)$auc.davis.goadrich
  ap    <- avgprec(p, y)
  # partial AUROC
  pauc  <- roc(response = y, predictor = p,
               partial.auc      = paucr, 
               partial.auc.focus= paucf)$auc[1]
  
  discperf <- data.frame("AUROC"     = cstat,
                         "AUPRC"     = auprc,
                         "AP"        = ap,
                         "pAUROC"    = pauc)
  return(discperf)
}

# 4.4) Calibration Metrics
# flexcal = "loess", "rcs3", veya "rcs5"
CalPerfBin <- function(y, p, flexcal = "loess", ngr = 10) {
  
  oe  <- sum(y) / sum(p)  # O:E oranı
  int <- coef(summary(glm(y ~ 1, offset = qlogis(p), family="binomial")))[1,1]
  sl  <- coef(summary(glm(y ~ qlogis(p), family="binomial")))[2,1]
  
  # Flexible calibration
  if (flexcal == "loess") {
    flc <- predict(loess(y ~ p, degree = 2))
  } else if (flexcal == "rcs3") {
    flc <- predict(glm(y ~ rms::rcs(qlogis(p), 3),
                       family="binomial"), type = "response")
  } else if (flexcal == "rcs5") {
    flc <- predict(glm(y ~ rms::rcs(qlogis(p), 5),
                       family="binomial"), type = "response")
  } else {
    stop("flexcal must be 'loess', 'rcs3', or 'rcs5'.")
  }
  
  eci <- mean((flc - p)^2) /
    mean((rep(mean(y), length(y)) - p)^2)
  ici <- mean(abs(flc - p))
  
  hlt <- ResourceSelection::hoslem.test(y, p, g = ngr)
  ece <- sum(abs(hlt$expected[,2] - hlt$observed[,2]) / length(y))
  
  data.frame("O:E ratio"     = oe,
             "Cal. intercept"= int,
             "Cal. slope"    = sl,
             "ECI"           = eci,
             "ICI"           = ici,
             "ECE"           = ece)
}

# 4.5) Overall Performance Metrics
OvPerfBin <- function(y, p) {
  lli <- sum(dbinom(y, prob = p, size = 1, log = TRUE))            # Loglike
  ll0 <- sum(dbinom(y, prob = rep(mean(y), length(y)), size = 1, log=TRUE))
  llo <- -lli  # logloss = -sum(loglik)
  
  br  <- mean((y - p)^2)                                           # Brier
  bss <- 1 - br / mean((y - mean(y))^2)  # Brier Skill Score
  
  mfr2 <- 1 - (lli / ll0)                                          # McFadden
  csr2 <- 1 - exp(2*(ll0 - lli)/length(y))                         # Cox-Snell
  nr2  <- csr2 / (1 - exp(2*ll0/length(y)))                        # Nagelkerke
  
  ds   <- mean(p[y==1]) - mean(p[y==0])                            # Disc. slope
  mape <- mean(abs(y - p))
  
  data.frame("Loglikelihood"    = lli,
             "Logloss"          = llo,
             "Brier"            = br,
             "Scaled Brier"     = bss,
             "McFadden R2"      = mfr2,
             "Cox-Snell R2"     = csr2,
             "Nagelkerke R2"    = nr2,
             "Discrimination slope" = ds,
             "MAPE"             = mape)
}

# 4.6) Classification Metrics (Sensitivity, Specificity, PPV, vb.)
ClassPerfBin <- function(y, p, cut = 0.5) {
  TP <- mean((p >= cut) & (y == 1))
  FN <- mean((p <  cut) & (y == 1))
  TN <- mean((p <  cut) & (y == 0))
  FP <- mean((p >= cut) & (y == 0))
  
  Sens <- TP / (TP + FN)
  Spec <- TN / (TN + FP)
  PPV  <- TP / (TP + FP)
  NPV  <- TN / (TN + FN)
  
  Acc  <- TP + TN
  Bar  <- 0.5 * (Sens + Spec)
  You  <- Sens + Spec - 1
  
  # Diagnostik odds oranı:
  # DOR = (Sens/(1-Spec)) / ((1-Sens)/Spec) 
  # if else'lara yakalanmaması için min. positivity check'ler eklenebilir
  DOR  <- (Sens / (1 - Spec)) / ((1 - Sens)/Spec)
  
  # Kappa
  Acc_E <- mean(y)*(TP + FP) + (1 - mean(y))*(FN + TN)
  Kap   <- (Acc - Acc_E) / (1 - Acc_E)
  
  # F1 ve MCC
  F1  <- 2 * ((PPV * Sens) / (PPV + Sens))
  MCC <- (TP*TN - FP*FN) / sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
  
  data.frame("Accuracy"         = Acc,
             "Balanced Accuracy"= Bar,
             "Youden index"     = You,
             "DOR"              = DOR,
             "Kappa"            = Kap,
             "F1"               = F1,
             "MCC"              = MCC,
             "Sensitivity/Recall" = Sens,
             "Specificity"      = Spec,
             "PPV/precision"    = PPV,
             "NPV"              = NPV)
}

# 4.7) Utility Metrics (Net Benefit, vb.)
UtilPerfBin <- function(y, p, cut = 0.5, costratio = 9) {
  
  # Net Benefit:
  NB  <- mean((p >= cut) & (y == 1)) - 
    (cut/(1 - cut)) * mean((p >= cut) & (y == 0))
  SNB <- NB / mean(y)
  
  # Expected Cost:
  risksort <- sort(p)
  ecpts <- data.frame(matrix(NA, nrow = length(risksort), ncol = 3))
  
  for (i in seq_along(risksort)) {
    ecFN <- sum(p[y==1] < risksort[i]) / sum(y==1)
    ecFP <- sum(p[y==0] >= risksort[i]) / sum(y==0)
    ecpts[i,3] <- ecFN * mean(y)   * (costratio * (costratio>1) + 1*(costratio<=1)) +
      ecFP * (1-mean(y)) * (costratio * (costratio<1) + 1*(costratio>=1))
  }
  EC          <- min(ecpts[,3])
  ECthreshold <- risksort[which.min(ecpts[,3])]
  
  data.frame("Net benefit"             = NB,
             "Standardized net benefit"= SNB,
             "Expected cost (EC)"      = EC,
             "Threshold for EC"        = ECthreshold)
}

# 4.8) EC Plot Helper Function
#  (Farklı normalized cost değerleri için minimum Expected Cost'u hesaplar.)
ecplotv <- function(y, p, ncostfp) {
  risksort <- sort(p)
  ec_res   <- data.frame(matrix(NA, nrow = 2, ncol = length(ncostfp)))
  
  for (i in seq_along(ncostfp)) {
    ecpts <- data.frame(matrix(NA, nrow = length(risksort), ncol = 3))
    for (j in seq_along(risksort)) {
      ecFN <- sum(p[y==1] < risksort[j]) / sum(y==1)
      ecFP <- sum(p[y==0] >= risksort[j]) / sum(y==0)
      # Normalized cost of FP = ncostfp[i], FN = 1 - ncostfp[i]
      ecpts[j,3] <- ecFN*mean(y)*(1 - ncostfp[i]) +
        ecFP*(1-mean(y))*ncostfp[i]
    }
    ec_res[1,i] <- min(ecpts[,3])                # min expected cost
    ec_res[2,i] <- risksort[which.min(ecpts[,3])]# threshold at min cost
  }
  return(ec_res)
}


############## örnek ############################

# Datasetin adını değiştirin
mydata <- read_csv("Desktop/statistics METASTATA/adem/adem/manuscript/merged_dataset.csv")

#"True_Labels", "Logistic_Regression","TabTransformer", "KAN", "TabNet","MLP"
mydata$TabTransformer <- ifelse(
  mydata$TabTransformer == 0,
  mydata$TabTransformer + 0.00000001,
  ifelse(
    mydata$TabTransformer == 1,
    mydata$TabTransformer - 0.00000001,
    mydata$TabTransformer
  )
)


mydata <- merged_dataset

# Değişken isimlerini değiştirin
library(dplyr)

#recalib öncesi
mydata <- mydata %>%
  rename(outcome_col = True_Labels, pred_col = TabTransformer)

outcome_col <- "outcome_col"
pred_col    <- "pred_col"


#recalib sonrası
# Logit dönüşümü ile rekalibrasyon
model <- glm(True_Labels ~ qlogis(Logistic_Regression), 
             data = mydata, 
             family = binomial)

# Rekalibre edilmiş tahminleri hesapla
mydata$MLP_recalib <- predict(model, data = mydata, type="response")

# Kolonları yeniden adlandır
mydata <- mydata %>%
  rename(outcome_col = True_Labels,
         pred_col = MLP_recalib)

# Referans kolonları tanımla
outcome_col <- "outcome_col"
pred_col    <- "pred_col"






# 1) Discrimination ölçütleri
disc_measures <- DiscPerfBin(mydata[[outcome_col]], mydata[[pred_col]])
disc_measures

# 2) Calibration ölçütleri
cal_measures <- CalPerfBin(mydata[[outcome_col]], mydata[[pred_col]], flexcal="loess")
cal_measures

# 3) Overall ölçütler
ov_measures  <- OvPerfBin(mydata[[outcome_col]], mydata[[pred_col]])
ov_measures

# 4) Classification ölçütleri (cut=0.10)
class_measures <- ClassPerfBin(mydata[[outcome_col]], mydata[[pred_col]], cut)
class_measures

# 5) Utility ölçütleri (cut=0.10, costratio=9)
util_measures  <- UtilPerfBin(mydata[[outcome_col]], mydata[[pred_col]], 
                              cut, costratio)
util_measures

# ------------ 6) PLOT ÖRNEKLERİ ------------ #
# 6.1) ROC ve PR noktaları hesaplamak, çizmek
# (manuel olarak da hesaplanabilir, ya da pROC / yardstick doğrudan kullanılabilir)

# ROC noktaları:
p_sorted <- sort(mydata[[pred_col]])
roc_pts  <- data.frame(Sens = numeric(length(p_sorted)), FPR = numeric(length(p_sorted)))
for (i in seq_along(p_sorted)) {
  roc_pts$Sens[i] <- sum(mydata[[pred_col]][mydata[[outcome_col]]==1] >= p_sorted[i]) /
    sum(mydata[[outcome_col]]==1)
  roc_pts$FPR[i]  <- sum(mydata[[pred_col]][mydata[[outcome_col]]==0] >= p_sorted[i]) /
    sum(mydata[[outcome_col]]==0)
}

# ROC plot:
ggplot(roc_pts, aes(x=FPR, y=Sens)) +
  geom_line() +
  geom_abline(slope = 1, intercept=0, linetype=2, color="gray") +
  theme_classic() +
  labs(x="1 - Specificity (FPR)", y="Sensitivity")

# 6.2) Kalibrasyon grafiği (val.prob.ci.2)
CalibrationCurves::val.prob.ci.2(
  p      = mydata[[pred_col]],
  y      = mydata[[outcome_col]],
  CL.smooth     = "fill",
  logistic.cal  = FALSE,
  g             = 10,
  col.ideal     = "gray",
  lty.ideal     = 2,
  lwd.ideal     = 2,
  dostats       = FALSE,
  col.log       = "gray",
  lty.log       = 1,
  lwd.log       = 2.5,
  col.smooth    = "black",
  lty.smooth    = 1,
  lwd.smooth    = 2.5,
  xlab          = "Estimated probability"
)

# 6.3) Karar eğrileri (Decision Curve Analysis, rmda paketi)
dca_model <- rmda::decision_curve(
  formula = as.formula(paste(outcome_col, "~", pred_col)), 
  data    = mydata,
  fitted.risk = TRUE,
  thresholds  = seq(0,1,0.01),
  confidence.intervals = FALSE
)

rmda::plot_decision_curve(
  dca_model,
  standardize = FALSE, 
  curve.names = "MyModel", 
  xlim = c(0,1), ylim=c(0,0.4),
  legend.position="topright"
)

# ------------ 7) BOOTSTRAP GÜVEN ARALIKLARI ------------ #
# Aşağıdaki örnek, 1000 tekrar ile neticeleri saklayıp basit yüzdeci (percentile) CI hesaplar.

set.seed(1234)
nboot <- 1000

results_mat <- matrix(NA, nrow = nboot, ncol = 34)
colnames(results_mat) <- c(
  # OvPerfBin (9), DiscPerfBin(4), CalPerfBin(6),
  # ClassPerfBin(11), UtilPerfBin(4) => 9+4+6+11+4=34
  colnames(OvPerfBin(mydata[[outcome_col]], mydata[[pred_col]])),
  colnames(DiscPerfBin(mydata[[outcome_col]], mydata[[pred_col]])),
  colnames(CalPerfBin(mydata[[outcome_col]], mydata[[pred_col]], flexcal="loess")),
  colnames(ClassPerfBin(mydata[[outcome_col]], mydata[[pred_col]], cut)),
  colnames(UtilPerfBin(mydata[[outcome_col]], mydata[[pred_col]], cut, costratio))
)

for (b in seq_len(nboot)) {
  idx <- sample(nrow(mydata), replace=TRUE)
  bootdata <- mydata[idx, ]
  
  ov_b    <- OvPerfBin(   bootdata[[outcome_col]], bootdata[[pred_col]])
  disc_b  <- DiscPerfBin( bootdata[[outcome_col]], bootdata[[pred_col]])
  cal_b   <- CalPerfBin(  bootdata[[outcome_col]], bootdata[[pred_col]], "loess")
  class_b <- ClassPerfBin(bootdata[[outcome_col]], bootdata[[pred_col]], cut)
  util_b  <- UtilPerfBin( bootdata[[outcome_col]], bootdata[[pred_col]], cut, costratio)
  
  results_mat[b, ] <- c(as.numeric(ov_b),
                        as.numeric(disc_b),
                        as.numeric(cal_b),
                        as.numeric(class_b),
                        as.numeric(util_b))
}

# Yüzdeci CI
ci_df <- data.frame(
  Estimate = c(as.numeric(OvPerfBin(mydata[[outcome_col]], mydata[[pred_col]])),
               as.numeric(DiscPerfBin(mydata[[outcome_col]], mydata[[pred_col]])),
               as.numeric(CalPerfBin(mydata[[outcome_col]], mydata[[pred_col]], "loess")),
               as.numeric(ClassPerfBin(mydata[[outcome_col]], mydata[[pred_col]], cut)),
               as.numeric(UtilPerfBin(mydata[[outcome_col]], mydata[[pred_col]], cut, costratio))),
  LCL = apply(results_mat, 2, quantile, probs=0.025),
  UCL = apply(results_mat, 2, quantile, probs=0.975)
)
row.names(ci_df) <- colnames(results_mat)
ci_df

# Artık ci_df tablosunda tüm metriklerin noktada tahminleri (Estimate) ve
# %95 güven aralıkları (LCL, UCL) yer alır.

######################
#  SON
######################
