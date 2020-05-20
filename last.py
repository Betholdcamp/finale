import sys
from pyspark import SparkContext
from collections import Counter
import re

if __name__ == '__main__':
    sc = SparkContext()
    streets = sc.textFile('/data/share/bdm/nyc_cscl.csv', use_unicode=False).cache() #'/data/share/bdm/complaints.csv'
    all_tickets = sc.textFile('/data/share/bdm/nyc_parking_violation/2015.csv,/data/share/bdm/nyc_parking_violation/2016.csv,/data/share/bdm/nyc_parking_violation/2017.csv,/data/share/bdm/nyc_parking_violation/2018.csv,/data/share/bdm/nyc_parking_violation/2019.csv', use_unicode=False).cache()
    
    def lines(partId, records):
        if partId==0:
            next(records) 
        
        import csv
        reader = csv.reader(records)
        next(reader)
        for row in reader:
            if row[0] != '' and row[2] != '' and row[3] != '' and row[4] != '' and row[5] != '':
                (PHYSICALID, L_LOW_HN, L_HIGH_HN, R_LOW_HN, R_HIGH_HN, ST_LABEL, BOROCODE, FULL_STREE) = (row[0], row[2], row[3], row[4], row[5], row[10], row[13], row[28])
                L_low1 = re.findall(r'\d+',L_LOW_HN)
                L_low = list(map(int, L_low1))
                L_low = tuple(L_low)
                
                L_high1 = re.findall(r'\d+',L_HIGH_HN)
                L_high = list(map(int, L_high1))
                L_high = tuple(L_high)
                
                R_low1 = re.findall(r'\d+',R_LOW_HN)
                R_low = list(map(int, R_low1))
                R_low = tuple(R_low)
                
                R_high1 = re.findall(r'\d+',R_HIGH_HN)
                R_high = list(map(int, R_high1))
                R_high = tuple(R_high)
                
                borocode = ('Unknown', 'NY' , 'BX', 'K', 'Q', 'R')
                yield (int(PHYSICALID), (L_low, L_high), (R_low, R_high), ST_LABEL, borocode[int(BOROCODE)], FULL_STREE)
    
    street_line = streets.mapPartitionsWithIndex(lines)
    street_list = sc.broadcast(street_line.collect())
    
    def findid(borough, street, h_num):
        dd = None
        for i in street_list.value:
            if (i[3] == street or i[5] == street) and (i[4]==borough) and ((h_num[-1] >= i[2][0][-1] and h_num[-1] <= i[2][1][-1]) or (h_num[-1] >= i[1][0][-1] and h_num[-1] <= i[1][1][-1])):
                    dd = i[0]
                    break
            else:
                dd = None
                break
        return dd
    
    def extractScores(partId, records):
        if partId==0:
            next(records)
        import csv
        reader = csv.reader(records)
        next(reader)
        for row in reader:
            if row[4] != '' and row[21] != '' and row[23] != '' and row[24] != '':
                (date, county, house_number, street_name, ) = (row[4], row[21], row[23], row[24])
                d = int(date.split('/')[-1])
                #d = date.split('/')
                temp = re.findall(r'\d+',house_number)
                res = list(map(int,temp))
                res = tuple(res)
                if len(res) > 0:
                    idd = findid(county, street_name, res)
                    if idd != None and (d >= 2015 and d <= 2019):
                        yield (idd, d)
    
    ticket = all_tickets.mapPartitionsWithIndex(extractScores)
    test = ticket.map(lambda x: (x[0], {x[1]: 1} )) \
    .reduceByKey(lambda x,y: (Counter(x) + Counter(y))) \
    .mapValues(lambda x: ([i for i in x.values()], len(x.keys()))) \
    .mapValues(lambda x: (tuple(x[0]), (x[0][-1]- x[0][0])/ x[1]))
    
    #test.saveAsTextFile('finale')
    final = test.map(lambda x: (x[0], x[1][0][0], x[1][0][1], x[1][0][2], x[1][0][3], x[1][0][4])) 
    
    sqlContext = SQLContext(sc)
    df = sqlContext.createDataFrame(final, ('ID', '2015_Count', '2016_Count', '2017_Count', '2018_Count', '2019_Count','OLS'))
    
    df.write \
    .format('com.databricks.spark.csv') \
    .option('header', 'true') \
    .save(sys.argv[1])
