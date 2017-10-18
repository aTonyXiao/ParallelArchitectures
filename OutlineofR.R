library(data.table)
library(parallel)

edges = fread("twitter_combined.txt", sep = " ", header = FALSE)

recippar <- function(edges) {
  nrow = nrow(edges)
  nthread = 4
  edges = edges[order(edges[,1], edges[,2])] # sort the data.table
  # cluster = makePSOCKcluster(rep("localhost", nthread))
  # chunk <- sample.int(nrow, nrow, replace = TRUE) # generate the random num for later each chunk
  count = 0
  for (i in 1:nrow(edges)) {
    found = edges[V1==edges[i]$V2 & V2==edges[i]$V1]
    if (nrow(found) > 0) {
      count = count + 1
    }
  }
  count
}

count = recippar(edges) / 2
print(count)
