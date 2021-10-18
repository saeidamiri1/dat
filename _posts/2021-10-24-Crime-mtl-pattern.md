---
title: "A Dive into Montreal Crime Pattern."
author: "Saeid Amiri"
date: '2021-10-24'
description: A Dive into Montreal Crime Pattern
layout: post
published: yes
tags: R leaf
categories: R_learn Practice
comments: no
---



## Introduction
Last year, someone broke our car's window parked inside the parking and took something from it, I though, it is good to look at the rubbing data in Montreal where we have been living about two years. 

Montreal is one of the AI hub city, and provides open data [open data](http://donnees.ville.montreal.qc.ca/dataset) to accessible for every one. Amazingly Montreal police has also been releasing detailed data, which can be used to explore this multicultural city. The analysis was completed using data from the [website](http://donnees.ville.montreal.qc.ca/) and R. Here, the analysis is done in Rmarkdown. The following analyses show the criminal hotspots and concentrations that has high high risk. 

## Reading data
Import data:


```r
# Load the data Montreal and prepare 
crime_mtl<- read.csv("https://data.montreal.ca/dataset/5829b5b0-ea6f-476f-be94-bc2b8797769a/resource/c6f482bf-bf0f-4960-8b2f-9982c211addd/download/interventionscitoyendo.csv", header = TRUE)

crime_mtl$CATEGORIE[crime_mtl$CATEGORIE=="Infractions entrainant la mort"]<-"resulting_in_death"
crime_mtl$CATEGORIE[crime_mtl$CATEGORIE=="Introduction"]<-"breaking_into_house"
crime_mtl$CATEGORIE[crime_mtl$CATEGORIE=="M\xe9fait"]<-"mischief"
crime_mtl$CATEGORIE[crime_mtl$CATEGORIE=="Vol dans / sur v\xe9hicule \xe0 moteur"]<-"theft_of_vehicle_part"
crime_mtl$CATEGORIE[crime_mtl$CATEGORIE=="Vol de v\xe9hicule \xe0 moteur"]<-"vehicle"
crime_mtl$CATEGORIE[crime_mtl$CATEGORIE=="Vols qualifi\xe9s"]<-"other types of robbery"
knitr::kable(crime_mtl[1:5,])
```



|CATEGORIE |DATE       |QUART | PDQ|        X|       Y| LONGITUDE| LATITUDE|
|:---------|:----------|:-----|---:|--------:|-------:|---------:|--------:|
|vehicle   |2018-09-13 |jour  |  30| 294904.2| 5047549| -73.62678| 45.56778|
|vehicle   |2018-04-30 |jour  |  30| 294904.2| 5047549| -73.62678| 45.56778|
|vehicle   |2018-09-01 |nuit  |   7| 290274.6| 5042150| -73.68593| 45.51912|
|mischief  |2017-07-21 |jour  |  21|      0.0|       0| -76.23729|  0.00000|
|mischief  |2017-07-29 |jour  |  12|      0.0|       0| -76.23729|  0.00000|

The data includes:  

- breaking_into_house(Fr: Introduction) : breaking and entering a public establishment or a private residence, theft of a firearm from a residence.
- theft_of_vehicle_part(Fr: Vol dans / sur véhicule à moteur) : theft of the contents of a motor vehicle (car, truck, motorcycle, etc.) or of a vehicle part (wheel, bumper, etc.).
- vehicle (Fr:Vol de véhicule à moteur) : car, truck, motorcycle theft, snowmobile tractor with or without trailer, construction or farm vehicle, all-terrain.
- mischie(Fr: Méfait): Graffiti and damage to religious property, vehicle or general damage and all other types of mischief.
Vol qualifié : Robbery accompanied by commercial violence, financial institution, person, handbag, armored vehicle, vehicle, firearm, and all other types of robbery.
- resulting_in_death(Fr: Infraction entraînant la mort): First degree murder, second degree murder, manslaughter, infanticide, criminal negligence, and all other types of offenses resulting in death.

## Pattern through years
It might also be interesting to see if the percentage changed through year, the following code provides the table of interest.  


|     | breaking_into_house|  mischief| other types of robbery| resulting_in_death| theft_of_vehicle_part|   vehicle|
|:----|-------------------:|---------:|----------------------:|------------------:|---------------------:|---------:|
|2015 |           0.1842427| 0.1812778|              0.1785289|          0.1525424|             0.1802642| 0.1412043|
|2016 |           0.1842615| 0.1606465|              0.1680572|          0.1299435|             0.1686045| 0.1390949|
|2017 |           0.1729767| 0.1584120|              0.1575856|          0.1468927|             0.1578200| 0.1512401|
|2018 |           0.1397243| 0.1395379|              0.1355355|          0.1751412|             0.1431214| 0.1354193|
|2019 |           0.1314112| 0.1338757|              0.1519666|          0.1355932|             0.1324195| 0.1332779|
|2020 |           0.1139198| 0.1273240|              0.1191044|          0.1468927|             0.1222626| 0.1504091|
|2021 |           0.0734639| 0.0989261|              0.0892219|          0.1129944|             0.0955078| 0.1493544|

Here we are interested in exploring the pattern of 
breaking and entering a public establishment or a private residence, so we drop unrelated crime. 


```r
crime_mtl_b <- subset(crime_mtl,CATEGORIE == "breaking_into_house")
crime_mtl_b<-crime_mtl_b[crime_mtl_b$LAT>1,]
```

The scatter plot over year shows there has been a decline which is a very good news. 


```r
suppressMessages(library(tidyverse))
suppressMessages(library(xts))
suppressMessages(library(lubridate))
suppressMessages(library(dygraphs)) 

crime_mtl_bj<-crime_mtl_b[crime_mtl_b$QUART=='jour',]
crime_mtl_bj_count <- crime_mtl_bj %>%
  group_by(DATE) %>% 
  summarise(count = n())

crime_mtl_bn<-crime_mtl_b[crime_mtl_b$QUART=='nuit',]
crime_mtl_bn_count <- crime_mtl_bn %>%
  group_by(DATE) %>% 
  summarise(count = n())

crime_mtl_bs<-crime_mtl_b[crime_mtl_b$QUART=='soir',]
crime_mtl_bs_count <- crime_mtl_bs %>%
  group_by(DATE) %>% 
  summarise(count = n())


crime_mtl_bjn_count<-merge(crime_mtl_bj_count,crime_mtl_bn_count,by.x="DATE", by.y="DATE",suffixes = c(".day",".night"))
crime_mtl_count<-merge(crime_mtl_bjn_count,crime_mtl_bs_count,by.x="DATE", by.y="DATE")

row.names(crime_mtl_count)=crime_mtl_count$DATE
crime_mtl_count<-crime_mtl_count[,-1]

dygraph(as.xts(crime_mtl_count)) %>%
  dySeries("count.day", label = "day") %>%
  dySeries("count.night", label = "night") %>%
  dySeries("count", label = "evening") 
```


<iframe src="https://github.com/saeidamiri1/dat/blob/main/public/Crimemtl/graph1.html" height="600" width="100%">
 </iframe>


## Spread of robbing
### Pattern over day


### Pattern over night


```r
LonLat<-crime_mtl_bn[,7:8]
kde <- bkde2D(LonLat,bandwidth=c(0.00225, 0.00225))
CL <- contourLines(kde$x1 , kde$x2 , kde$fhat,nlevels = 8)

LEVS <- as.factor(sapply(CL, `[[`, "level"))
NLEV <- length(levels(LEVS))

pgons <- lapply(1:length(CL), function(i)
  Polygons(list(Polygon(cbind(CL[[i]]$x, CL[[i]]$y))), ID=i))
spgons = SpatialPolygons(pgons)

im<-leaflet(spgons) %>% addTiles() %>%
  addPolygons(color = heat.colors(NLEV, NULL)[LEVS]) %>%
  addRectangles(lng1=min(LonLat[,1]), lat1=min(LonLat[,2]),
                lng2=max(LonLat[,1]), lat2=max(LonLat[,2]),
                fillColor = "transparent")

im
```

### Pattern over evening


```r
LonLat<-crime_mtl_bs[,7:8]
kde <- bkde2D(LonLat,bandwidth=c(0.00225, 0.00225))
CL <- contourLines(kde$x1 , kde$x2 , kde$fhat,nlevels = 8)

LEVS <- as.factor(sapply(CL, `[[`, "level"))
NLEV <- length(levels(LEVS))

pgons <- lapply(1:length(CL), function(i)
  Polygons(list(Polygon(cbind(CL[[i]]$x, CL[[i]]$y))), ID=i))
spgons = SpatialPolygons(pgons)

im<-leaflet(spgons) %>% addTiles() %>%
  addPolygons(color = heat.colors(NLEV, NULL)[LEVS]) %>%
  addRectangles(lng1=min(LonLat[,1]), lat1=min(LonLat[,2]),
                lng2=max(LonLat[,1]), lat2=max(LonLat[,2]),
                fillColor = "transparent")

im
```
