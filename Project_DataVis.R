#Data Visualization Project

#setwd("/home/ai16/R_Assignments/Data")
setwd("D:/Projects")
agri_prod_df=read.csv("TEST_Prod-Dep_of_Agri_and_Coop.csv",head=TRUE)
agri_prod_df

#============
#barplot
#============

filter_agri_prod_df<-agri_prod_df[which(agri_prod_df$Particulars == 'Foodgrains' | agri_prod_df$Particulars == 'Foodgrains Kharif' | agri_prod_df$Particulars == 'Foodgrains Rabi'),4:13]
filter_agri_prod_df

matrx=data.matrix(filter_agri_prod_df, rownames.force = NA)
matrx

barplot(matrx,beside=TRUE,
        main="Agriculture Production in India (2005-2014)",
        xlab="Year ->",ylab="In Million Tonnes ->",col=rainbow(3),legend.text=c("Total Prod.","Kharif","Rabi"))


#============
#barplot - Statewise yield (2013)
#============
state_agri_prod_df<-agri_prod_df[grep('^Foodgrains Yield ',agri_prod_df$Particulars),c(1,12)]
state_agri_prod_df

barplot(state_agri_prod_df)

matrx=data.matrix(state_agri_prod_df, rownames.force = NA,stringsAsFactors =FALSE)
matrx


barplot(matrx,beside=TRUE,
        main="State wise Agriculture Production (2013)",
        xlab="States ->",ylab="In Million Tonnes ->",col=rainbow(17)) #,legend.text=c("Total Prod.","Kharif","Rabi")



#=============================
#Histogram of area under prod
#=============================

Area<-agri_prod_df[grep('^Foodgrains Area ',agri_prod_df$Particulars),12]
Area
hist(Area,xlab="Area under cultivation (in miln. hec.)->",
     main="Statewise area under cultivation (Year-2013)",
     ylab="Count of States ->", col=rainbow(4))

#=============================#=============================
#Normal Distribution for Statewise Yeild for a given year
#=============================#=============================

mean1<-mean(agri_prod_df[grep('^Foodgrains Yield *',agri_prod_df$Particulars),12],na.rm=TRUE)

sd1<-sd(agri_prod_df[grep('^Foodgrains Yield *',agri_prod_df$Particulars),12],na.rm=TRUE)

x <- seq(-4,4,length=100)*sd1 + mean1
hx <- dnorm(x,mean1,sd1)

plot(x, hx, type="p", xlab="Yield Per Hec ->", main="Yield Per Hec. (Year-2013)" ) #axes=FALSE, ylab="test"

#=================================
#Box Plot (Target Prod ~ Achieved)
#=================================

dummy_target<-data.frame(type=rep("Target",8))

target_prod<-data.frame(agri_prod_df[grep('^Major Crops Target*',agri_prod_df$Particulars),c(1,12)])

target_dtls<-cbind(dummy_target,target_prod)

dummy_achiev<-data.frame(type=rep("Achieved",8))

achiev_prod<-data.frame(agri_prod_df[grep('^Major Crops Achievements*',agri_prod_df$Particulars),c(1,12)])

achiev_dtls<-cbind(dummy_achiev,achiev_prod)

prod_consoli<-rbind(target_dtls,achiev_dtls)

boxplot(prod_consoli$X2013~prod_consoli$type,
        ylab="Prodn. in Million tonnes ->",xlab="Category",
        main="Targeted Vs Achieved Food Prod. (Year-2013)",col=rainbow(2))


