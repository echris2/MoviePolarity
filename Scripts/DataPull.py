# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 18:59:41 2020

@author: eachr
"""

import os

import sqlite3
import csv




### inverse normal dist, tends to the extremes
## high score / low score ratio

def movieCSVcur(db_loc,statement, output):
    
    from Scripts import db_ec
    conn = db_ec.connect_db(db_loc)
    cur = conn.cursor()
 
    cur.execute(statement)
    with open(output, 'w', encoding = 'utf8') as myfile:
        writer = csv.writer(myfile)
        writer.writerow(i[0] for i in cur.description)
        writer.writerows(cur)

   
    conn.close()
   


