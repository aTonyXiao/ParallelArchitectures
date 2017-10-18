library(data.table)
library(parallel)

recippar <- function(edges) {
  cluster = makeCluster(2)
  combined = parLapply(cluster, list(cbind(edges$V1, edges$V2), cbind(edges$V2, edges$V1)), merge_edges)
  edges$V3 = combined[1]
  edges$V4 = combined[2]
  # edges$V3 = paste(list[,1], list[,2])
  # edges$V4 = paste(list2[,1], list2[,2])
  # print(list)
  # print(list[1][1])
  # combined =
  print("After")
  # edges$V3 = combined[1]
  # edges$V4 = combined[2]
  # print("Middle")
  # second = cbind(edges$V1, edges$V2)
  # edges$V4 = clusterApply(cluster, second, combine)
  # print("After")
  # edges$V4 = clusterApply(cluster, edges, reversed_combine)
  # barr()
  found = intersect(edges$V3, edges$V4)
  same = edges[edges$V1 == edges$V2]
  (length(found) - nrow(same)) / 2
}

merge_edges <- function(edge) {
  # edge[,1]
  paste(edge[,1], edge[,2])
}

# reversed_combine <- function(edge) {
#   edge$V1 + edge$V2
# }

# setReverseArray <- function(edges, local_edges) {
#   require(Parallel)
#   local_edges <- splitIndices(nrow(edges), myinfo$nwrkrs)[[4]]
#   local_edges$V3 = paste(local_edges$V1, local_edges$V2)
#   local_edges$V4 = paste(local_edges$V2, local_edges$V1)
#   invisible(0)
# }

edges = fread("twitter_combined.txt", sep = " ", header = FALSE)

start = as.numeric(Sys.time())
# count = clusterEvalQ(cluster, recippar(edges))
count = recippar(edges)
end = as.numeric(Sys.time())

print(count)
print(end - start)
