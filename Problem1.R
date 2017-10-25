library(data.table)
library(parallel)

recippar <- function(edges) {
  cluster = makeCluster(2)
  combined = parLapply(cluster, list(cbind(edges$V1, edges$V2), cbind(edges$V2, edges$V1)), merge_edges)
  stopCluster(cluster)
  edges$V3 = combined[1]
  edges$V4 = combined[2]
  found = length(intersect(edges$V3, edges$V4))
  repeated = nrow(edges[edges$V1 == edges$V2]) # e.g. 20 20 is repeated, 20 21 is not.
  (found - repeated) / 2
}

merge_edges <- function(edge) {
  paste(edge[,1], edge[,2])
}
