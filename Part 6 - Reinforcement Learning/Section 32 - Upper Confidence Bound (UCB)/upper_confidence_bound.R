# Importing dataset
dataset = read.csv("Ads_CTR_Optimisation.csv")

# Implementing UCB
N = nrow(dataset)
d = ncol(dataset)
ads_selected = vector()
numbers_of_selections = integer(d)
sums_of_rewards = integer(d)
total_reward = 0
for (n in 1:N){
  max_upper_bound = 0
  ad = 0
  for (i in 1:d){
    if (numbers_of_selections[i] > 0){
      average_reward = sums_of_rewards[i] / numbers_of_selections[i]
      delta_i = sqrt(1.5 * log(n) / numbers_of_selections[i])
      upper_bound = average_reward + delta_i
    }
    else{
      # nas oprimeira interações força cada uma das ads a ter o maximo
      # upper_bound e ser eleita 
      upper_bound = 1e400
    }
    if (upper_bound > max_upper_bound){
      max_upper_bound = upper_bound
      ad = i
    }
  }
  ads_selected = c(ads_selected, ad)
  numbers_of_selections[ad] = numbers_of_selections[ad] + 1
  reward = dataset[n,ad]
  sums_of_rewards[ad] = sums_of_rewards[ad] + reward
  total_reward = total_reward + reward
}

# Visualizing the results
hist(ads_selected,
     col = "blue",
     main = "Histogram of ads selections",
     xlab = "Ads",
     ylab = "Number of times each ad was selected")