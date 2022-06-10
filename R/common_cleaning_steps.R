###################################################
#Step 1: loading packages and data
###################################################

#loading packages
library(dplyr) #for data maniupulation
library(lubridate) #for easy conversion of dates

#loading data
df <- read.csv("employment-data.csv")

###################################################
#Step 2: basic data exploration
###################################################

#view columns names
colnames(df)
#view columns names and data types
str(df)
#see counts of values within a variable
table(df$Period)
#see unique of values within a variable
unique(df$Group)

###################################################
#Step 3: cleaning data
###################################################

#date conversion (see ymd function documentation)
#convert column from number to character to date 
df <- df %>% mutate(Period = ym(as.character(Period)))
#filter df on conditions
df <- df %>% filter(Series_title_2 == "Agriculture, Forestry and Fishing" |
                      Series_title_2 == "Mining")
#select columns
df <- df %>% select(c("Period", "Data_value", "Subject", "Group", 
                      "Series_title_1", "Series_title_2"))
#drop column
df <- df %>% select(-"Series_title_1")
#add calculated columns 
df <- df %>% mutate(Data_value_sqrt = sqrt(Data_value))

#in one chain
df <- read.csv("employment-data.csv")

df <- df %>% mutate(Period = ym(as.character(Period))) %>%
  filter(Series_title_2 == "Agriculture, Forestry and Fishing" |
           Series_title_2 == "Mining") %>%
  select(c("Period", "Data_value", "Subject", "Group", 
           "Series_title_1", "Series_title_2")) %>%
  select(-"Series_title_1") %>% #irl you'd just not select this column in the line above
  mutate(Data_value_sqrt = sqrt(Data_value))

#remove rows with null values 
df <- na.omit(df)
#remove columns containing only  null values
df<- Filter(function(x)!all(is.na(x)), df)

###################################################
#Step 4: aggregating data
###################################################

dfSummary <- df %>%
  group_by(Series_title_2) %>% #grouping column(s)
  summarise()




