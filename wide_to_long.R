###################################################
#Step 1: loading packages and data
###################################################

#loading packages
library(tidyr) #for data maniupulation
library(dplyr)

#loading data
df <- read.csv("batting_averages.csv")

###################################################
#Step 2: Pivot from wide to long
###################################################

#pivot wide to long, with team names becaome a column "opponent"
df <- df %>% pivot_longer(!Player, names_to = "opponent", values_to = "batting_average")
  