library(data.table)
library(parallel)

recippar <- function(edges) {
  edges$V3 = paste(edges$V1, edges$V2)
  edges$V4 = paste(edges$V2, edges$V1)
  found = intersect(edges$V3, edges$V4)
  same = edges[edges$V1 == edges$V2]
  (length(found) - nrow(same)) / 2
}

edges = fread("twitter_combined.txt", sep = " ", header = FALSE)
start = as.numeric(Sys.time())
count = recippar(edges)
end = as.numeric(Sys.time())

print(count)
print(end - start)
