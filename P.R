library(data.table)
library(parallel)
nthread = 4
count = 0
cluster = makePSOCKcluster(rep("localhost", nthread))
data = fread("test1.txt", sep = " ", header = FALSE)

smaller_than <- function(lhs, rhs) {
	#both are pair
	if (lhs[1] == rhs[1])
		return (lhs[2] < rhs[2])
	else
		return (lhs[1] < rhs[1])
}

binary_search <- function (edges, value) {
	up = nrow(edges)
	down = 1
	
	while(down <= up) {
		midpoint = ceiling((up+down)/2)
		pair = as.numeric(edges[midpoint])
		if (smaller_than(value, pair))
			up = midpoint - 1
		else if (smaller_than(pair, value))
			down = midpoint + 1
		else
			return (midpoint)
	}
	
	return (-1)
}

recippar <- function (edges) {
	data = edges[order(V1, V2),]
	for(x in 1:nrow(data)) {
		
		found <- binary_search(data, rev(as.numeric(data[x])))
		if(found != -1) {
			count = count + 1
		}
	}		
	count / 2
}

recippar(data)
