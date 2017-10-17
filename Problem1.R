recippar <- function (edges) {
	maxNum = 20
	matrix = list()
	matrix[[maxNum]] = NA
	count = 0
	total = 0
	for (i in 1:nrow(edges)) {
		pairs = as.numeric(edges[i])	
		first = pairs[1]
		second = pairs[2]
		matrix[[first]] = c(second, matrix[[first]])
	}
	
	separate <- function(cls, matrix) {
		rowgrps = splitIndices(length(matrix), length(cls))
		local <- function (rows) {
			for (r in rows)
				for (i in matrix[[r]])
					if (r %in% matrix[[i]])
						count = count + 1
			count
		}

		ans = clusterApply(cluster, rowgrps, local)
		for (res in ans)
			total = total + sum(res)
		total / 2
	}
	
	separate(cluster, matrix)
}

library(data.table)
library(parallel)	
nthread = 4
cluster = makePSOCKcluster(rep("localhost", nthread))
data = fread("test1.txt", sep = " ", header = FALSE)
recippar(data)
