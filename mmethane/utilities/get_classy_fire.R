library(Biostrings); packageVersion("Biostrings")
library(ggplot2); packageVersion("ggplot2")
library(plyr); packageVersion("plyr")
library(openxlsx)
library(pracma)
library(classyfireR)
args = commandArgs(trailingOnly=TRUE)
print(args[1])
print(args[2])
if (length(args)==0) {
  stop("At least one argument must be supplied (input file).n", call.=FALSE)
  # temp <- read.csv(paste("./datasets/v_microbiome_TEST/tmp", "/mtp_map_wINchiKey.csv", sep=""))
  } else {
  temp <- read.csv(paste(args[1], args[2], sep=""))
}
# stop("At least one argument must be supplied (input file).n", call.=FALSE)}
# temp<- read.csv("./datasets/IBMDB/processed/ibmdb_pubchem/metabolite_InChIKey_only.csv")
# temp <- temp[(1:11),]

cmpd_list <- temp[,2]
cmpd_names <-temp[,1]
class_list <- c()
cmpds <- c()
df_list <- list()
for (row in 1:nrow(temp)){
  cmpd <- temp[row, 2]
  name <- temp[row, 1]
  if (cmpd!=""){
    Classification <- get_classification(cmpd)
    if (!is.null(Classification)){
      df <- as.data.frame(classification(Classification)[,1:2])
      colnames(df)[1] <- 'Level'
      colnames(df)[2] <- name
      df_list[[row]] <- df
      class_list <- c(class_list, Classification)
      cmpds <- c(name)
    }
  }
}
# df_list <- df_list[-509]
df_list[sapply(df_list, is.null)] <- NULL
tab <- Reduce(function(...) merge(..., by = 'Level', all = T), df_list)
rownames(tab) <- tab$Level
tab_sort <- tab[c("kingdom","superclass", "class", "subclass", "level 5", "level 6", "level 7", "level 8", "level 9", "level 10"),]
tab_sort$Level <- NULL
write.table(tab_sort, file = paste(args[1],"classy_fire_df.csv",sep=""), sep = ',', row.names = TRUE, col.names = NA)
