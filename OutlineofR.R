library(data.table)
library(parallel)
edges = fread("twitter_combined.txt", sep = " ", header = FALSE)

recippar(edges)

recippar <- function(edges){
  sorted <- edges[order(edges[,1],edges[,2])]#sort the data.table
  nrow = nrow(edges)
  count = 0
  nthread = 4
  cluster = makePSOCKcluster(rep("localhost", nthread))
  chunk <- sample.int(nrow, nrow, replace = TRUE)#generate the random num for later each chunk

}
 #order the data frame


# between(new, lower, upper, incbounds=TRUE)




# edge <- split(data, as.factor(data))

# recippar(data)
#


# recippar <- function (edges) {
#decreasing sorted
#
#   for(x in 1:length(data))
#     found <- binary_search(data, data[x], index=TRUE)
#     if(found)
#       count = count + 1
#
#   count / 2
# }
