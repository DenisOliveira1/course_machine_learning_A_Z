# Importing dataset
dataset = read.csv("Ads_CTR_Optimisation.csv")

# Implementing Random Selection
N = nrow(dataset)
d = ncol(dataset)
ads_selected = vector()
total_reward = 0
for (n in 1:N){
  ad = sample(d, 1)
  ads_selected = c(ads_selected, ad)
  reward = dataset[n,ad]
  total_reward = total_reward + reward
}

# Visualizing the results
hist(ads_selected,
     col = "blue",
     main = "Histogram of ads selections",
     xlab = "Ads",
     ylab = "Number of times each ad was selected")