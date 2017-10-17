library(data.table)
library(parallel)
nthread = 4
count = 0
cluster = makePSOCKcluster(rep("localhost", nthread))
data = fread("test1.txt", sep = "\n", header = FALSE)
as.numeric(data)

recippar(data)

recippar <- function (edges) {
  sortedArray <- parApply(cluster,data,1,sort.int, method='quick'))#decreasing sorted

  for(x in 1:length(data))
    found <- binary_search(data, data[x], index=TRUE)
    if(found)
      count = count + 1

  count / 2
}
