directory <- file.choose()
DIR <- dirname(directory)
setwd(DIR)

movie_meat <- read.csv("../Data/movies_metadata_clean.csv", header = TRUE)
ratings_dat <- read.csv("../Data/ratings.csv", header = TRUE)

links <-read.csv("../Data/links.csv", header = TRUE) 

##plot histogram of revenues and budgets
library(ggplot2)
g <- ggplot(movie_meat[movie_meat$budget > 0,], aes(x = budget))
g2 <- ggplot(movie_meat[movie_meat$budget > 0,], aes(x = log(budget)))

g3 <- ggplot(movie_meat[movie_meat$budget > 0,], aes(x = log(budget), y = log(revenue)))

fivenum(movie_meat$budget)
summary(movie_meat$revenue)

g + geom_histogram(binwidth = 9000000, fill = "lightblue", color = "black") + theme_minimal() + labs(title = "Budgets [where greater than 0]",
                                                                                                     x = "Budget Dollars", y = "Count")

g2 + geom_histogram(fill = "lightblue", color = "black") + theme_minimal() + labs(title = "Budgets [where greater than 0]",
                                                                                  x = "log(Budget Dollars)", y = "Count")

g3 + geom_point() + theme_minimal() 
##plot number of ratings

summary(movie_meat$vote_count)

gRats <- ggplot(movie_meat, aes(x = vote_count))
logRats <- ggplot(movie_meat[movie_meat$vote_count > 0,], aes(x = log(vote_count)))

gRats + geom_histogram(fill = "lightblue", color = "black") + theme_minimal() + labs(title = "Number of Votes per Movie",
                                                                                                     x = "Votes", y = "Count")
logRats + geom_histogram(fill = "lightblue", color = "black") + theme_minimal() + labs(title = "Log(Vote_Count)",
                                                                                     x = "log(Votes)", y = "Count")

#plot average movie ratings

summary(movie_meat$vote_average)

gAvgs <- ggplot(movie_meat[movie_meat$vote_count > 0,], aes(x = vote_average))
gAvgs + geom_histogram(fill = "lightblue", color = "black") + theme_minimal() + labs(title = "Average Movie Ratings",
                                                                                     x = "Average Rating", y = "Count")
##plot barplot of user ratings

aggRatings <- read.csv("../Data/RatingsCount.csv")
gUratings <- ggplot(aggRatings,aes(x = rating, y = rCount/1000))
gUratings + geom_bar(stat = "identity", fill = "lightblue", color = "black") + theme_minimal() + labs(title = "User Ratings", x = "Rating",
                                                                 y = "Rating Count (`000)") + 
  scale_y_continuous(label=scales::comma) + theme(axis.text.x = element_text(size = 10),axis.text.y = element_text(size = 10))



##PROBS
aggRatings
totCount <- sum(aggRatings$rCount)
badCount <- sum(aggRatings$rCount[aggRatings$rating <= 1.0])
goodCount <- sum(aggRatings$rCount[aggRatings$rating > 4.0])

ProbBad <- round(badCount/totCount,4)
ProbGood <- round(goodCount/totCount,4)

#Good/Bad Ratio
baseRatio <- ProbGood/ProbBad



##Controverisla 

##Twilight 8966 id, 50620, 50619, 23921,18239

library(dplyr)

#Battlefield earth 5491
#Twilight 8966
#The Avengers 24428
#Shawshank Redemption 278

ids <- c(5491,8966,24428,278)

link_all <- filter(links, tmdbId %in% ids)
ratings_all <- filter(ratings_dat, movieId %in% link_all$movieId)

gg <- ggplot(ratings_all, aes(x = rating, group = movieId))


gg + geom_histogram(fill = "lightblue", color = "black") + theme_minimal() + labs(title = "Number of Votes per Movie",
                                                                                     x = "Votes", y = "Count") +
  facet_grid(~movieId)

movie_hist <- function(movid){
  title <- movie_meat$title[movie_meat$id == movid]
  year <- movie_meat$release_date[movie_meat$id == movid]
  link <- filter(links, tmdbId == movid)
  rats <- filter(ratings_dat, movieId == link$movieId[1])
  
  g <- ggplot(rats, aes(x = rating))
  
  gg <- g + geom_bar(stat = "count",fill = "lightblue", color = "black") + theme_minimal() + labs(title = paste0("\"",title,"\""," Rating Distribution"),
                                                                                   x = "Rating", y = "Count") +
    geom_vline(aes(xintercept = mean(rating)),col='red',size=2, linetype = 'dashed') +
    scale_y_continuous(label=scales::comma) + theme(axis.text.x = element_text(size = 14),axis.text.y = element_text(size = 16),
                                                    axis.title.x = element_text(face = "bold"), axis.title.y = element_text(face = "bold"),
                                                    plot.title = element_text(size = 20, face = "bold.italic"))
  
  print(gg)
  return(rats)
}

movie_meat[grep("Last Temptation",movie_meat$title),c("id", "title", "release_date")]
movie_meat[grep("Twilight",movie_meat$title),c("id", "title", "release_date", "vote_average", "vote_count")]


movie_hist(5491)#Battlefield Earth

movie_hist(24428)#,"The Avengers")
movie_hist(278)#,"The Shawshank Redemption")

movie_hist(11051)#,"Last Temptation of Christ, 1988")
movie_hist(19846)#,"The Interview, 2014")
movie_hist(591)#,"The Da Vinci")

tdat <- movie_hist(597)
tprop <- prop.table(table(tdat$rating))

alohadat <- movie_hist(222936)#Aloha
passiondat <- movie_hist(615)#,"Passion of the Christ, 2004")
twidat <- movie_hist(8966)#,"Twilight")
hostdat <- movie_hist(1690)#Hostel

gamedat <- movie_hist(18501)#Gamer
battdat <- movie_hist(5491)#Battlefield Earth

avgdat <- movie_hist(24428)#,"The Avengers")
avgProp <- prop.table(table(avgdat$rating))
shawdat <- movie_hist(278)#,"The Shawshank Redemption")
shawProp <- prop.table(table(shawdat$rating))

gameProp <- prop.table(table(gamedat$rating)) 
battProp <- prop.table(table(battdat$rating))

ratioizer <- function(propTable){
  lowclass <- sum(propTable[1:2])
  hiclass <- sum(propTable[9:10])
  ratio <- hiclass/lowclass
  HiLowProp <- lowclass + hiclass
  return(c(ratio, HiLowProp))
}





alProp <- prop.table(table(alohadat$rating))

lowclass <- sum(passProp[1:2])
hiclass <- sum(passProp[9:10])




passProp <- prop.table(table(passiondat$rating))
twiProp <- prop.table(table(twidat$rating))
hostProp <- prop.table(table(hostdat$rating))

ratioizer(alProp)
ratioizer(passProp)
ratioizer(twiProp)
ratioizer(hostProp)

avgPRop <- prop.table(table(hostdat$rating))
              

dats <- sapply(ratings_dat[,'timestamp'],as.POSIXct, origin = "1970-01-01")
