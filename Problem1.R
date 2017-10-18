library(data.table)
library(parallel)

recippar <- function(edges) {
  require(parallel)
  local_edges <- splitIndices(nrow(edges), myinfo$nwrkrs)[[myinfo$id]]
  local_edges$V3 = paste(local_edges$V1, local_edges$V2)
  local_edges$V4 = paste(local_edges$V2, local_edges$V1)
  barr()
  found = intersect(edges$V3, edges$V4)
  same = edges[edges$V1 == edges$V2]
  (length(found) - nrow(same)) / 2
}

edges = fread("twitter_combined.txt", sep = " ", header = FALSE)

cluster = makecluster(rep("localhost", 4))
mgrinit(cluster)
clusterEvalQ(cluster, id <- myinfo$id)
clusterExport(cluster, "recippar")

start = as.numeric(Sys.time())
count = clusterEvalQ(cluster, recippar(edges))
end = as.numeric(Sys.time())

print(count)
print(end - start)
