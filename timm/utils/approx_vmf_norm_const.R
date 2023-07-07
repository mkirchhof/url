# This script compares the true log(cp(kappa)) normalizing factor to approximations from the literature

# True Cp(k)
logcp_ratio = function(kappa, p = 512){
  # Calculate for the big kappa
  res = (p/2 - 1) * log(kappa) - p/2 * log(2 * pi) - log(besselI(kappa, p / 2 - 1, expon.scaled = TRUE)) - kappa
  res[res == Inf | res == -Inf] = NA
  # Calculate for kappa = 0
  res0 = -log(2) - p/2 * log(pi) + lgamma(p / 2)
  res0[res0 == Inf | res0 == -Inf] = NA
  return(res0 - res)
}

# Taylor approximation
# Values generated for kappa: 10:500
p = 128
kappa = 10:500
gt = logcp_ratio(kappa, p)
approx = lm(y ~ x + I(x^1.55), data=data.frame(y=gt, x=kappa))
summary(approx)
taylor = predict(approx, data.frame(x=kappa))
plot(kappa[!is.na(taylor)], predict(approx, newdata=data.frame(x=kappa)), xlim=range(kappa), type = "l", las=1, #main=expression("Taylor Approximation of"~log(C[p](kappa))),
     xlab="", ylab="", col="red")
points(x = kappa[!is.na(gt)], y = gt[!is.na(gt)], col="black", type="l")
title(sub=paste0(c("p =", p), collapse=" "))
print(summary(approx))


# Just the normalizing constant (for ELK)
logcp = function(kappa, p=10){
  res = (p/2 - 1) * log(kappa) - p/2 * log(2 * pi) - log(besselI(kappa, p / 2 - 1, expon.scaled = TRUE)) - kappa
  res[res == Inf | res == -Inf] = NA
  return(res)
}

p = 10
kappa = 10:500
gt = logcp(kappa, p=p)
approx = lm(y ~ x + I(x^1.1), data=data.frame(y=gt, x=kappa))
summary(approx)
taylor = predict(approx, data.frame(x=kappa))
plot(kappa[!is.na(taylor)], predict(approx, newdata=data.frame(x=kappa)), xlim=range(kappa), type = "l", las=1, #main=expression("Taylor Approximation of"~log(C[p](kappa))),
     xlab="", ylab="", col="red")
points(x = kappa[!is.na(gt)], y = gt[!is.na(gt)], col="black", type="l")
title(sub=paste0(c("p =", p), collapse=" "))
print(summary(approx))