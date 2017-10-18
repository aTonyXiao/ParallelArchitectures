library(data.table)
library(parallel)

recippar <- function(edges) {

  cluster = makeCluster(4)
  mgrinit(cluster)
  mgrmakevar(cluster, "local_edges", nrow(edges), 4)
  clusterEvalQ(cluster, id <- myinfo$id)
  clusterExport(cluster, "setReverseArray")
  clusterEvalQ(cluster, setReverseArray(edges2, local_edges))
  barr()
  found = intersect(edges$V3, edges$V4)
  same = edges[edges$V1 == edges$V2]
  (length(found) - nrow(same)) / 2
}


setReverseArray <- function(edges, local_edges) {
  require(parallel)
  local_edges <- splitIndices(nrow(edges), myinfo$nwrkrs)[[myinfo$id]]
  local_edges$V3 = paste(local_edges$V1, local_edges$V2)
  local_edges$V4 = paste(local_edges$V2, local_edges$V1)
  invisible(0)
}

edges = fread("twitter_combined.txt", sep = " ", header = FALSE)

start = as.numeric(Sys.time())
# count = clusterEvalQ(cluster, recippar(edges))
recippar(edges)
end = as.numeric(Sys.time())

print(count)
print(end - start)
