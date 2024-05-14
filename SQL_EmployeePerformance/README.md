---
title: "SQL EmployeeRecords"
author: "Jas Ahuja"
date: "2024-05-14"
output:
  html_document:
    keep_md: yes
    code_folding: show
    toc: true
    toc_depth: 3
    toc_float: true
    theme: readable
    css: style.css
  pdf_document:
    extra_dependencies: ["adjustbox"]
    toc: true
    toc_depth: 3
email: jasvinderahuja@gmail.com
---
## Introduction

I am mostly self-taught or taught many years ago.

So I am refreshing and documenting my skills in a systematic manner and this SQL course is a part of that attempt. It covers most use cases for a non data engineer, where data retrieval is inolved. In future, I plan to add better explanations and more material. Stay tuned! 

_need to add Introductory chapter and joins_

_css chunk to make tables fit_

<style type="text/css">
#TOC {
  margin: 3px 0px 3px 3px;
}

td{font-size: 12px;}
</style>

_Hidden code chunk for mypassword_


_Make a MySQL connection from Rmarkdown_


```r
library(DBI)
library(dplyr)
library(dbplyr)

## mypassword = {}!

conn <- dbConnect(RMySQL::MySQL(),
                  dbname = "",
                  Server = "localhost",
                  port = 3306,
                  user = "root",
                  password = mypassword)

knitr::opts_chunk$set(connection = "conn", echo = TRUE, comment = NA, message = FALSE, warning = FALSE)
```


# Course-end Project 1 - ScienceQtech Employee Performance Mapping...

## Description

ScienceQtech is a startup that works in the Data Science field. ScienceQtech has worked on fraud detection, market basket, self-driving cars, supply chain, algorithmic early detection of lung cancer, customer sentiment, and the drug discovery field. With the annual appraisal cycle around the corner, the HR department has asked you (Junior Database Administrator) to generate reports on employee details, their performance, and on the project that the employees have undertaken, to analyze the employee database and extract specific data based on different requirements.

 

## Objective: 

To facilitate a better understanding, managers have provided ratings for each employee which will help the HR department to finalize the employee performance mapping. As a DBA, you should find the maximum salary of the employees and ensure that all jobs are meeting the organization's profile standard. You also need to calculate bonuses to find extra cost for expenses. This will raise the overall performance of the organization by ensuring that all required employees receive training.

__Note: You must download the dataset from the course resource section in LMS and create a table to perform the above objective.__

## Dataset description:

### emp_record_table: It contains the information of all the employees.
  - EMP_ID – ID of the employee
  - FIRST_NAME – First name of the employee
  - LAST_NAME – Last name of the employee
  - GENDER – Gender of the employee
  - ROLE – Post of the employee
  - DEPT – Field of the employee
  - EXP – Years of experience the employee has
  - COUNTRY – Country in which the employee is presently living
  - CONTINENT – Continent in which the country is
  - SALARY – Salary of the employee
  - EMP_RATING – Performance rating of the employee
  - MANAGER_ID – The manager under which the employee is assigned 
  - PROJ_ID – The project on which the employee is working or has worked on

### Proj_table: It contains information about the projects.
  - PROJECT_ID – ID for the project
  - PROJ_Name – Name of the project
  - DOMAIN – Field of the project
  - START_DATE – Day the project began
  - CLOSURE_DATE – Day the project was or will be completed
  - DEV_QTR – Quarter in which the project was scheduled
  - STATUS – Status of the project currently

### Data_science_team: It contains information about all the employees in the Data Science team.

  - EMP_ID – ID of the employee
  - FIRST_NAME – First name of the employee
  - LAST_NAME – Last name of the employee
  - GENDER – Gender of the employee
  - ROLE – Post of the employee
  - DEPT – Field of the employee
  - EXP – Years of experience the employee has
  - COUNTRY – Country in which the employee is presently living
  - CONTINENT – Continent in which the country is

## The tasks to be performed: 

### 1. Create a database named employee, then import tables into the employee database from the given resources.
     - data_science_team.csv 
     - proj_table.csv and 
     - emp_record_table.csv


__Answer-1:__ _Done: Using the import function of MySQLWorkbench._ 

> set nocount on  
> ! This works with SQL database to run multiple queries from a code chunk. This does not work with MySQL and therefore I need to use one query per code chunk          



1. Initiate use of our database.   


```sql
use employee
```
2. Look at tables available in _employee_:  


```sql
SHOW TABLES;
```


<div class="knitsql-table">


Table: 3 records

|Tables_in_employee |
|:------------------|
|data_science_team  |
|emp_record_table   |
|proj_table         |

</div>
3. Fix _the proj_table_   
- Alter table as text feild _project_id_ cannot be __PRIMARY KEY__    


```sql
ALTER TABLE proj_table
	MODIFY COLUMN project_id VARCHAR(10)
```


```sql
ALTER TABLE proj_table
	ADD PRIMARY KEY(project_id)
```
    
- Our altered table  
    

```sql
SELECT * FROM proj_table
```


<div class="knitsql-table">


Table: 6 records

|project_id |PROJ_NAME                      |DOMAIN     |START _DATE |CLOSURE_DATE |DEV_QTR |STATUS  |
|:----------|:------------------------------|:----------|:-----------|:------------|:-------|:-------|
|P103       |Drug Discovery                 |HEALTHCARE |04-06-2021  |6/20/2021    |Q1      |DONE    |
|P105       |Fraud Detection                |FINANCE    |04-11-2021  |6/25/2021    |Q1      |DONE    |
|P109       |Market Basket Analysis         |RETAIL     |04-12-2021  |6/30/2021    |Q1      |DELAYED |
|P204       |Supply Chain Management        |AUTOMOTIVE |07/15/2021  |9/28/2021    |Q2      |WIP     |
|P302       |Early Detection of Lung Cancer |HEALTHCARE |10-08-2021  |12/18/2021   |Q3      |YTS     |
|P406       |Customer Sentiment Analysis    |RETAIL     |07-09-2021  |9/24/2021    |Q2      |WIP     |

</div>
  
4. Fix _emp_record_table_   
- Lets look at the table    
      

```sql
SELECT * FROM emp_record_table
```


<div class="knitsql-table">


Table: Displaying records 1 - 10

|emp_id |first_name |LAST_NAME |GENDER |ROLE                  |DEPT       | EXP|COUNTRY |CONTINENT     | SALARY| EMP_RATING|MANAGER_ID |proj_id |designation |
|:------|:----------|:---------|:------|:---------------------|:----------|---:|:-------|:-------------|------:|----------:|:----------|:-------|:-----------|
|E001   |Arthur     |Black     |M      |PRESIDENT             |ALL        |  20|USA     |NORTH AMERICA |  16500|          5|NA         |NA      |NA          |
|E005   |Eric       |Hoffman   |M      |LEAD DATA SCIENTIST   |FINANCE    |  11|USA     |NORTH AMERICA |   8500|          3|E103       |P105    |NA          |
|E010   |William    |Butler    |M      |LEAD DATA SCIENTIST   |AUTOMOTIVE |  12|FRANCE  |EUROPE        |   9000|          2|E428       |P204    |NA          |
|E052   |Dianna     |Wilson    |F      |SENIOR DATA SCIENTIST |HEALTHCARE |   6|CANADA  |NORTH AMERICA |   5500|          5|E083       |P103    |NA          |
|E057   |Dorothy    |Wilson    |F      |SENIOR DATA SCIENTIST |HEALTHCARE |   9|USA     |NORTH AMERICA |   7700|          1|E083       |P302    |NA          |
|E083   |Patrick    |Voltz     |M      |MANAGER               |HEALTHCARE |  15|USA     |NORTH AMERICA |   9500|          5|E001       |NA      |NA          |
|E103   |Emily      |Grove     |F      |MANAGER               |FINANCE    |  14|CANADA  |NORTH AMERICA |  10500|          4|E001       |NA      |NA          |
|E204   |Karene     |Nowak     |F      |SENIOR DATA SCIENTIST |AUTOMOTIVE |   8|GERMANY |EUROPE        |   7500|          5|E428       |P204    |NA          |
|E245   |Nian       |Zhen      |M      |SENIOR DATA SCIENTIST |RETAIL     |   6|CHINA   |ASIA          |   6500|          2|E583       |P109    |NA          |
|E260   |Roy        |Collins   |M      |SENIOR DATA SCIENTIST |RETAIL     |   7|INDIA   |ASIA          |   7000|          3|E583       |NA      |NA          |

</div>

- Alter the table and put _emp_id_ as __PRIMARY_KEY__    
      

```sql
ALTER TABLE emp_record_table
	MODIFY COLUMN emp_id VARCHAR(10)
```
    

```sql
ALTER TABLE emp_record_table
	ADD PRIMARY KEY(emp_id)
```
  
- _proj_id_ needs to be set as __FOREIGN_KEY__     
- it cannot be NA or Null!    |      
- feild type also needs to match in the two tables     


```sql
SELECT * FROM emp_record_table
WHERE proj_id = 'NA' OR proj_id IS NULL
```


<div class="knitsql-table">


Table: 7 records

|emp_id |first_name |LAST_NAME |GENDER |ROLE                  |DEPT       | EXP|COUNTRY  |CONTINENT     | SALARY| EMP_RATING|MANAGER_ID |proj_id |designation |
|:------|:----------|:---------|:------|:---------------------|:----------|---:|:--------|:-------------|------:|----------:|:----------|:-------|:-----------|
|E001   |Arthur     |Black     |M      |PRESIDENT             |ALL        |  20|USA      |NORTH AMERICA |  16500|          5|NA         |NA      |NA          |
|E083   |Patrick    |Voltz     |M      |MANAGER               |HEALTHCARE |  15|USA      |NORTH AMERICA |   9500|          5|E001       |NA      |NA          |
|E103   |Emily      |Grove     |F      |MANAGER               |FINANCE    |  14|CANADA   |NORTH AMERICA |  10500|          4|E001       |NA      |NA          |
|E260   |Roy        |Collins   |M      |SENIOR DATA SCIENTIST |RETAIL     |   7|INDIA    |ASIA          |   7000|          3|E583       |NA      |NA          |
|E428   |Pete       |Allen     |M      |MANAGER               |AUTOMOTIVE |  14|GERMANY  |EUROPE        |  11000|          4|E001       |NA      |NA          |
|E583   |Janet      |Hale      |F      |MANAGER               |RETAIL     |  14|COLOMBIA |SOUTH AMERICA |  10000|          2|E001       |NA      |NA          |
|E612   |Tracy      |Norris    |F      |MANAGER               |RETAIL     |  13|INDIA    |ASIA          |   8500|          4|E001       |NA      |NA          |

</div>
- Fix the NA values in _proj_id_  

>In case of Error: 1175- go to menu "MySQLWorkbench" > "Settings" > "SQL Editor" > uncheck "Safe Updates"_  
>_on PC it is under "edit"_  


```sql

UPDATE emp_record_table
	SET proj_id=NULL
    WHERE proj_id='NA'
```
    
- Now change type to match the proj_table.   
    
    

```sql

ALTER TABLE emp_record_table
	MODIFY COLUMN proj_id VARCHAR(10)
```


```sql
ALTER TABLE emp_record_table
	ADD CONSTRAINT fk_proj
    FOREIGN KEY(proj_id) REFERENCES proj_table(PROJECT_ID)
```


- Let's see the altered table:  



```sql
DESCRIBE emp_record_table
```


<div class="knitsql-table">


Table: Displaying records 1 - 10

|Field      |Type        |Null |Key |Default |Extra |
|:----------|:-----------|:----|:---|:-------|:-----|
|emp_id     |varchar(10) |NO   |PRI |NA      |      |
|first_name |varchar(20) |YES  |MUL |NA      |      |
|LAST_NAME  |text        |YES  |    |NA      |      |
|GENDER     |text        |YES  |    |NA      |      |
|ROLE       |text        |YES  |    |NA      |      |
|DEPT       |text        |YES  |    |NA      |      |
|EXP        |int         |YES  |    |NA      |      |
|COUNTRY    |text        |YES  |    |NA      |      |
|CONTINENT  |text        |YES  |    |NA      |      |
|SALARY     |int         |YES  |    |NA      |      |

</div>

5. Similarly fix the _data_science_team_.   

```sql
SELECT * FROM data_science_team
```


<div class="knitsql-table">


Table: Displaying records 1 - 10

|emp_id |FIRST_NAME |LAST_NAME |GENDER |ROLE                     |DEPT       | EXP|COUNTRY  |CONTINENT     |
|:------|:----------|:---------|:------|:------------------------|:----------|---:|:--------|:-------------|
|E005   |Eric       |Hoffman   |M      |LEAD DATA SCIENTIST      |FINANCE    |  11|USA      |NORTH AMERICA |
|E010   |William    |Butler    |M      |LEAD DATA SCIENTIST      |AUTOMOTIVE |  12|FRANCE   |EUROPE        |
|E052   |Dianna     |Wilson    |F      |SENIOR DATA SCIENTIST    |HEALTHCARE |   6|CANADA   |NORTH AMERICA |
|E057   |Dorothy    |Wilson    |F      |SENIOR DATA SCIENTIST    |HEALTHCARE |   9|USA      |NORTH AMERICA |
|E204   |Karene     |Nowak     |F      |SENIOR DATA SCIENTIST    |AUTOMOTIVE |   8|GERMANY  |EUROPE        |
|E245   |Nian       |Zhen      |M      |SENIOR DATA SCIENTIST    |RETAIL     |   6|CHINA    |ASIA          |
|E260   |Roy        |Collins   |M      |SENIOR DATA SCIENTIST    |RETAIL     |   7|INDIA    |ASIA          |
|E403   |Steve      |Hoffman   |M      |ASSOCIATE DATA SCIENTIST |FINANCE    |   4|USA      |NORTH AMERICA |
|E478   |David      |Smith     |M      |ASSOCIATE DATA SCIENTIST |RETAIL     |   3|COLOMBIA |SOUTH AMERICA |
|E505   |Chad       |Wilson    |M      |ASSOCIATE DATA SCIENTIST |HEALTHCARE |   5|CANADA   |NORTH AMERICA |

</div>

- match emp_id to emp_record_table  


```sql
ALTER TABLE data_science_team
	MODIFY COLUMN emp_id VARCHAR(10)
```


```sql
ALTER TABLE data_science_team
	ADD CONSTRAINT fk_emp_record_table_emp_id
	FOREIGN KEY(emp_id) REFERENCES emp_record_table(emp_id)
```
  
  
### 2. Create an ER diagram for the given employee database.    
__Answer-2:__ goto "database" > "Reverse Engineer" and follow prompts to get __Figure-1__

![Figure-1: The Entity Relation diagram generated from MySQLWorkbench](EmployeePerformance.png)  
 

### 3. Write a query to fetch EMP_ID, FIRST_NAME, LAST_NAME, GENDER, and DEPARTMENT from the employee record table, and make a list of employees and details of their department.  


```sql

SELECT emp_id, first_name, last_name, gender, dept
	FROM emp_record_table

```


<div class="knitsql-table">


Table: Displaying records 1 - 10

|emp_id |first_name |last_name |gender |dept       |
|:------|:----------|:---------|:------|:----------|
|E001   |Arthur     |Black     |M      |ALL        |
|E005   |Eric       |Hoffman   |M      |FINANCE    |
|E010   |William    |Butler    |M      |AUTOMOTIVE |
|E052   |Dianna     |Wilson    |F      |HEALTHCARE |
|E057   |Dorothy    |Wilson    |F      |HEALTHCARE |
|E083   |Patrick    |Voltz     |M      |HEALTHCARE |
|E103   |Emily      |Grove     |F      |FINANCE    |
|E204   |Karene     |Nowak     |F      |AUTOMOTIVE |
|E245   |Nian       |Zhen      |M      |RETAIL     |
|E260   |Roy        |Collins   |M      |RETAIL     |

</div>
 

### 4. Write a query to fetch EMP_ID, FIRST_NAME, LAST_NAME, GENDER, DEPARTMENT, and EMP_RATING if the EMP_RATING is:   

-__4.1 _emp_rating_ less than two__  


```sql

SELECT emp_id, first_name, last_name, gender, dept
	FROM emp_record_table
    WHERE emp_rating < 2

```


<div class="knitsql-table">


Table: 3 records

|emp_id |first_name |last_name |gender |dept       |
|:------|:----------|:---------|:------|:----------|
|E057   |Dorothy    |Wilson    |F      |HEALTHCARE |
|E532   |Claire     |Brennan   |F      |AUTOMOTIVE |
|E620   |Katrina    |Allen     |F      |RETAIL     |

</div>

-__4.2 _emp_rating_ greater than four__    


```sql

SELECT emp_id, first_name, last_name, gender, dept
	FROM emp_record_table
    WHERE emp_rating > 4

```


<div class="knitsql-table">


Table: 4 records

|emp_id |first_name |last_name |gender |dept       |
|:------|:----------|:---------|:------|:----------|
|E001   |Arthur     |Black     |M      |ALL        |
|E052   |Dianna     |Wilson    |F      |HEALTHCARE |
|E083   |Patrick    |Voltz     |M      |HEALTHCARE |
|E204   |Karene     |Nowak     |F      |AUTOMOTIVE |

</div>

-__4.3 _emp_rating_ Between two and four__  


```sql

SELECT emp_id, first_name, last_name, gender, dept
	FROM emp_record_table
    WHERE emp_rating BETWEEN 2 AND 4

```


<div class="knitsql-table">


Table: Displaying records 1 - 10

|emp_id |first_name |last_name |gender |dept       |
|:------|:----------|:---------|:------|:----------|
|E005   |Eric       |Hoffman   |M      |FINANCE    |
|E010   |William    |Butler    |M      |AUTOMOTIVE |
|E103   |Emily      |Grove     |F      |FINANCE    |
|E245   |Nian       |Zhen      |M      |RETAIL     |
|E260   |Roy        |Collins   |M      |RETAIL     |
|E403   |Steve      |Hoffman   |M      |FINANCE    |
|E428   |Pete       |Allen     |M      |AUTOMOTIVE |
|E478   |David      |Smith     |M      |RETAIL     |
|E505   |Chad       |Wilson    |M      |HEALTHCARE |
|E583   |Janet      |Hale      |F      |RETAIL     |

</div>

-__4.4 _emp_rating_ bins__    


```sql

SELECT emp_id, first_name, last_name, gender, dept
	FROM emp_record_table
    WHERE emp_rating < 2 OR emp_rating BETWEEN 2 AND 4 OR emp_rating > 4

```


<div class="knitsql-table">


Table: Displaying records 1 - 10

|emp_id |first_name |last_name |gender |dept       |
|:------|:----------|:---------|:------|:----------|
|E001   |Arthur     |Black     |M      |ALL        |
|E005   |Eric       |Hoffman   |M      |FINANCE    |
|E010   |William    |Butler    |M      |AUTOMOTIVE |
|E052   |Dianna     |Wilson    |F      |HEALTHCARE |
|E057   |Dorothy    |Wilson    |F      |HEALTHCARE |
|E083   |Patrick    |Voltz     |M      |HEALTHCARE |
|E103   |Emily      |Grove     |F      |FINANCE    |
|E204   |Karene     |Nowak     |F      |AUTOMOTIVE |
|E245   |Nian       |Zhen      |M      |RETAIL     |
|E260   |Roy        |Collins   |M      |RETAIL     |

</div>


### 5. Write a query to concatenate the FIRST_NAME and the LAST_NAME of employees in the Finance department from the employee table and then give the resultant column alias as NAME.  


```sql
SELECT CONCAT(first_name, ",", last_name) AS NAME FROM emp_record_table
	WHERE dept='FINANCE'
```


<div class="knitsql-table">


Table: 3 records

|NAME          |
|:-------------|
|Eric,Hoffman  |
|Emily,Grove   |
|Steve,Hoffman |

</div>

### 6. Write a query to list only those employees who have someone reporting to them. Also, show the number of reporters (including the President).  


```sql
SELECT * FROM emp_record_table
	WHERE emp_id IN (SELECT DISTINCT manager_id from emp_record_table)
```


<div class="knitsql-table">


Table: 6 records

|emp_id |first_name |LAST_NAME |GENDER |ROLE      |DEPT       | EXP|COUNTRY  |CONTINENT     | SALARY| EMP_RATING|MANAGER_ID |proj_id |designation |
|:------|:----------|:---------|:------|:---------|:----------|---:|:--------|:-------------|------:|----------:|:----------|:-------|:-----------|
|E103   |Emily      |Grove     |F      |MANAGER   |FINANCE    |  14|CANADA   |NORTH AMERICA |  10500|          4|E001       |NA      |NA          |
|E428   |Pete       |Allen     |M      |MANAGER   |AUTOMOTIVE |  14|GERMANY  |EUROPE        |  11000|          4|E001       |NA      |NA          |
|E083   |Patrick    |Voltz     |M      |MANAGER   |HEALTHCARE |  15|USA      |NORTH AMERICA |   9500|          5|E001       |NA      |NA          |
|E001   |Arthur     |Black     |M      |PRESIDENT |ALL        |  20|USA      |NORTH AMERICA |  16500|          5|NA         |NA      |NA          |
|E583   |Janet      |Hale      |F      |MANAGER   |RETAIL     |  14|COLOMBIA |SOUTH AMERICA |  10000|          2|E001       |NA      |NA          |
|E612   |Tracy      |Norris    |F      |MANAGER   |RETAIL     |  13|INDIA    |ASIA          |   8500|          4|E001       |NA      |NA          |

</div>

### 7. Write a query to list down all the employees from the healthcare and finance departments using union. Take data from the employee record table.  


```sql
SELECT * FROM emp_record_table
	WHERE dept = 'HEALTHCARE'
UNION
SELECT * FROM emp_record_table
	WHERE dept = 'FINANCE'
```


<div class="knitsql-table">


Table: 7 records

|emp_id |first_name |LAST_NAME |GENDER |ROLE                     |DEPT       | EXP|COUNTRY |CONTINENT     | SALARY| EMP_RATING|MANAGER_ID |proj_id |designation |
|:------|:----------|:---------|:------|:------------------------|:----------|---:|:-------|:-------------|------:|----------:|:----------|:-------|:-----------|
|E052   |Dianna     |Wilson    |F      |SENIOR DATA SCIENTIST    |HEALTHCARE |   6|CANADA  |NORTH AMERICA |   5500|          5|E083       |P103    |NA          |
|E057   |Dorothy    |Wilson    |F      |SENIOR DATA SCIENTIST    |HEALTHCARE |   9|USA     |NORTH AMERICA |   7700|          1|E083       |P302    |NA          |
|E083   |Patrick    |Voltz     |M      |MANAGER                  |HEALTHCARE |  15|USA     |NORTH AMERICA |   9500|          5|E001       |NA      |NA          |
|E505   |Chad       |Wilson    |M      |ASSOCIATE DATA SCIENTIST |HEALTHCARE |   5|CANADA  |NORTH AMERICA |   5000|          2|E083       |P103    |NA          |
|E005   |Eric       |Hoffman   |M      |LEAD DATA SCIENTIST      |FINANCE    |  11|USA     |NORTH AMERICA |   8500|          3|E103       |P105    |NA          |
|E103   |Emily      |Grove     |F      |MANAGER                  |FINANCE    |  14|CANADA  |NORTH AMERICA |  10500|          4|E001       |NA      |NA          |
|E403   |Steve      |Hoffman   |M      |ASSOCIATE DATA SCIENTIST |FINANCE    |   4|USA     |NORTH AMERICA |   5000|          3|E103       |P105    |NA          |

</div>
 

### 8. Write a query to list down employee details such as EMP_ID, FIRST_NAME, LAST_NAME, ROLE, DEPARTMENT, and EMP_RATING grouped by dept. Also include the respective employee rating along with the max emp rating for the department.    


```sql

SELECT emp_id, first_name, last_name, role, dept, MAX(emp_rating) AS max_emp_rating
	FROM emp_record_table
-- this GROUP BY is used for max_emp_rating
	GROUP BY emp_id, first_name, last_name, role, dept

```


<div class="knitsql-table">


Table: Displaying records 1 - 10

|emp_id |first_name |last_name |role                  |dept       | max_emp_rating|
|:------|:----------|:---------|:---------------------|:----------|--------------:|
|E001   |Arthur     |Black     |PRESIDENT             |ALL        |              5|
|E005   |Eric       |Hoffman   |LEAD DATA SCIENTIST   |FINANCE    |              3|
|E010   |William    |Butler    |LEAD DATA SCIENTIST   |AUTOMOTIVE |              2|
|E052   |Dianna     |Wilson    |SENIOR DATA SCIENTIST |HEALTHCARE |              5|
|E057   |Dorothy    |Wilson    |SENIOR DATA SCIENTIST |HEALTHCARE |              1|
|E083   |Patrick    |Voltz     |MANAGER               |HEALTHCARE |              5|
|E103   |Emily      |Grove     |MANAGER               |FINANCE    |              4|
|E204   |Karene     |Nowak     |SENIOR DATA SCIENTIST |AUTOMOTIVE |              5|
|E245   |Nian       |Zhen      |SENIOR DATA SCIENTIST |RETAIL     |              2|
|E260   |Roy        |Collins   |SENIOR DATA SCIENTIST |RETAIL     |              3|

</div>
 

### 9. Write a query to calculate the minimum and the maximum salary of the employees in each role. Take data from the employee record table.  


```sql
SELECT ROLE, MIN(salary), MAX(salary) 
	FROM emp_record_table
    GROUP BY ROLE
```


<div class="knitsql-table">


Table: 6 records

|ROLE                     | MIN(salary)| MAX(salary)|
|:------------------------|-----------:|-----------:|
|PRESIDENT                |       16500|       16500|
|LEAD DATA SCIENTIST      |        8500|        9000|
|SENIOR DATA SCIENTIST    |        5500|        7700|
|MANAGER                  |        8500|       11000|
|ASSOCIATE DATA SCIENTIST |        4000|        5000|
|JUNIOR DATA SCIENTIST    |        2800|        3000|

</div>

### 10. Write a query to assign ranks to each employee based on their experience. Take data from the employee record table.  


```sql
SELECT *, RANK() OVER (ORDER by exp DESC)
	FROM emp_record_table
```


<div class="knitsql-table">


Table: Displaying records 1 - 10

|emp_id |first_name |LAST_NAME |GENDER |ROLE                  |DEPT       | EXP|COUNTRY  |CONTINENT     | SALARY| EMP_RATING|MANAGER_ID |proj_id |designation | RANK() OVER (ORDER by exp DESC)|
|:------|:----------|:---------|:------|:---------------------|:----------|---:|:--------|:-------------|------:|----------:|:----------|:-------|:-----------|-------------------------------:|
|E001   |Arthur     |Black     |M      |PRESIDENT             |ALL        |  20|USA      |NORTH AMERICA |  16500|          5|NA         |NA      |NA          |                               1|
|E083   |Patrick    |Voltz     |M      |MANAGER               |HEALTHCARE |  15|USA      |NORTH AMERICA |   9500|          5|E001       |NA      |NA          |                               2|
|E103   |Emily      |Grove     |F      |MANAGER               |FINANCE    |  14|CANADA   |NORTH AMERICA |  10500|          4|E001       |NA      |NA          |                               3|
|E428   |Pete       |Allen     |M      |MANAGER               |AUTOMOTIVE |  14|GERMANY  |EUROPE        |  11000|          4|E001       |NA      |NA          |                               3|
|E583   |Janet      |Hale      |F      |MANAGER               |RETAIL     |  14|COLOMBIA |SOUTH AMERICA |  10000|          2|E001       |NA      |NA          |                               3|
|E612   |Tracy      |Norris    |F      |MANAGER               |RETAIL     |  13|INDIA    |ASIA          |   8500|          4|E001       |NA      |NA          |                               6|
|E010   |William    |Butler    |M      |LEAD DATA SCIENTIST   |AUTOMOTIVE |  12|FRANCE   |EUROPE        |   9000|          2|E428       |P204    |NA          |                               7|
|E005   |Eric       |Hoffman   |M      |LEAD DATA SCIENTIST   |FINANCE    |  11|USA      |NORTH AMERICA |   8500|          3|E103       |P105    |NA          |                               8|
|E057   |Dorothy    |Wilson    |F      |SENIOR DATA SCIENTIST |HEALTHCARE |   9|USA      |NORTH AMERICA |   7700|          1|E083       |P302    |NA          |                               9|
|E204   |Karene     |Nowak     |F      |SENIOR DATA SCIENTIST |AUTOMOTIVE |   8|GERMANY  |EUROPE        |   7500|          5|E428       |P204    |NA          |                              10|

</div>

_NB: Lets understand this one-_  
- _SELECT *: This selects all columns from the emp_record_table._  
- _RANK() OVER (ORDER by exp DESC): This is a window function that calculates the rank of each record based on the exp column in descending order._  
- _The ORDER BY exp DESC specifies that the records should be ranked based on the exp column in descending order_  

### 11. Write a query to create a view that displays employees in various countries whose salary is more than six thousand. Take data from the employee record table.  
_Can't have two views by the same name!_  

```sql

DROP VIEW IF EXISTS vw_country

```


```sql

CREATE VIEW vw_country
	AS
    SELECT first_name, last_name, country FROM emp_record_table
    WHERE salary > 6000
    
```

### 12. Write a nested query to find employees with experience of more than ten years. Take data from the employee record table.  


```sql
SELECT * FROM
(
  SELECT * FROM emp_record_table
	  WHERE exp > 10
) AS T

```


<div class="knitsql-table">


Table: 8 records

|emp_id |first_name |LAST_NAME |GENDER |ROLE                |DEPT       | EXP|COUNTRY  |CONTINENT     | SALARY| EMP_RATING|MANAGER_ID |proj_id |designation |
|:------|:----------|:---------|:------|:-------------------|:----------|---:|:--------|:-------------|------:|----------:|:----------|:-------|:-----------|
|E001   |Arthur     |Black     |M      |PRESIDENT           |ALL        |  20|USA      |NORTH AMERICA |  16500|          5|NA         |NA      |NA          |
|E005   |Eric       |Hoffman   |M      |LEAD DATA SCIENTIST |FINANCE    |  11|USA      |NORTH AMERICA |   8500|          3|E103       |P105    |NA          |
|E010   |William    |Butler    |M      |LEAD DATA SCIENTIST |AUTOMOTIVE |  12|FRANCE   |EUROPE        |   9000|          2|E428       |P204    |NA          |
|E083   |Patrick    |Voltz     |M      |MANAGER             |HEALTHCARE |  15|USA      |NORTH AMERICA |   9500|          5|E001       |NA      |NA          |
|E103   |Emily      |Grove     |F      |MANAGER             |FINANCE    |  14|CANADA   |NORTH AMERICA |  10500|          4|E001       |NA      |NA          |
|E428   |Pete       |Allen     |M      |MANAGER             |AUTOMOTIVE |  14|GERMANY  |EUROPE        |  11000|          4|E001       |NA      |NA          |
|E583   |Janet      |Hale      |F      |MANAGER             |RETAIL     |  14|COLOMBIA |SOUTH AMERICA |  10000|          2|E001       |NA      |NA          |
|E612   |Tracy      |Norris    |F      |MANAGER             |RETAIL     |  13|INDIA    |ASIA          |   8500|          4|E001       |NA      |NA          |

</div>
 

### 13. Write a query to create a _stored procedure_ to retrieve the details of the employees whose experience is more than three years. Take data from the employee record table.   
- Need to drop procedure if exists  


```sql

DROP PROCEDURE IF EXISTS sp_exp;
```


```sql

CREATE PROCEDURE sp_exp()
BEGIN
    SELECT * FROM emp_record_table WHERE exp > 3;
END;

```


__In MySQLWorkbench we need a delimiter change. The code would therefore be-__
  
> DELIMITER //  
>  
>CREATE PROCEDURE sp_exp()  
>BEGIN  
>    SELECT * FROM emp_record_table WHERE exp > 3;  
>END//
>  
>DELIMITER ;  



_Here, // is used as the delimiter. It's changed back to ; after the stored procedure definition._  

_This ensures that the SQL interpreter doesn't interpret the semicolons within the stored procedure as statement terminators until the entire procedure is defined._  


### 14. Write a query using stored functions in the project table to check whether the job profile assigned to each employee in the data science team matches the organization’s set standard.  
The standard being:  
- For an employee with experience less than or equal to 2 years assign 'JUNIOR DATA SCIENTIST',  
- For an employee with the experience of 2 to 5 years assign 'ASSOCIATE DATA SCIENTIST',  
- For an employee with the experience of 5 to 10 years assign 'SENIOR DATA SCIENTIST',  
- For an employee with the experience of 10 to 12 years assign 'LEAD DATA SCIENTIST',  
- For an employee with the experience of 12 to 16 years assign 'MANAGER'.  


```sql
SELECT * ,
	CASE WHEN exp <= 2 THEN 'JUNIOR DATA SCIENTIST'
    WHEN exp > 2 AND EXP  <= 5 THEN 'ASSOCIATE DATA SCIENTIST'
    WHEN exp > 5 AND EXP  <= 10 THEN 'SENIOR DATA SCIENTIST'
    WHEN exp > 10 AND EXP  <= 12 THEN 'LEAD DATA SCIENTIST'
    WHEN exp > 12 AND EXP  <= 16 THEN 'MANAGER'
    END AS designation
FROM emp_record_table

```


<div class="knitsql-table">


Table: Displaying records 1 - 10

|emp_id |first_name |LAST_NAME |GENDER |ROLE                  |DEPT       | EXP|COUNTRY |CONTINENT     | SALARY| EMP_RATING|MANAGER_ID |proj_id |designation |designation           |
|:------|:----------|:---------|:------|:---------------------|:----------|---:|:-------|:-------------|------:|----------:|:----------|:-------|:-----------|:---------------------|
|E001   |Arthur     |Black     |M      |PRESIDENT             |ALL        |  20|USA     |NORTH AMERICA |  16500|          5|NA         |NA      |NA          |NA                    |
|E005   |Eric       |Hoffman   |M      |LEAD DATA SCIENTIST   |FINANCE    |  11|USA     |NORTH AMERICA |   8500|          3|E103       |P105    |NA          |LEAD DATA SCIENTIST   |
|E010   |William    |Butler    |M      |LEAD DATA SCIENTIST   |AUTOMOTIVE |  12|FRANCE  |EUROPE        |   9000|          2|E428       |P204    |NA          |LEAD DATA SCIENTIST   |
|E052   |Dianna     |Wilson    |F      |SENIOR DATA SCIENTIST |HEALTHCARE |   6|CANADA  |NORTH AMERICA |   5500|          5|E083       |P103    |NA          |SENIOR DATA SCIENTIST |
|E057   |Dorothy    |Wilson    |F      |SENIOR DATA SCIENTIST |HEALTHCARE |   9|USA     |NORTH AMERICA |   7700|          1|E083       |P302    |NA          |SENIOR DATA SCIENTIST |
|E083   |Patrick    |Voltz     |M      |MANAGER               |HEALTHCARE |  15|USA     |NORTH AMERICA |   9500|          5|E001       |NA      |NA          |MANAGER               |
|E103   |Emily      |Grove     |F      |MANAGER               |FINANCE    |  14|CANADA  |NORTH AMERICA |  10500|          4|E001       |NA      |NA          |MANAGER               |
|E204   |Karene     |Nowak     |F      |SENIOR DATA SCIENTIST |AUTOMOTIVE |   8|GERMANY |EUROPE        |   7500|          5|E428       |P204    |NA          |SENIOR DATA SCIENTIST |
|E245   |Nian       |Zhen      |M      |SENIOR DATA SCIENTIST |RETAIL     |   6|CHINA   |ASIA          |   6500|          2|E583       |P109    |NA          |SENIOR DATA SCIENTIST |
|E260   |Roy        |Collins   |M      |SENIOR DATA SCIENTIST |RETAIL     |   7|INDIA   |ASIA          |   7000|          3|E583       |NA      |NA          |SENIOR DATA SCIENTIST |

</div>
 
- If we were to add a column like this -   

```sql
ALTER TABLE emp_record_table
	ADD designation VARCHAR(50)
```


```sql
UPDATE emp_record_table
	SET designation=CASE WHEN exp <= 2 THEN 'JUNIOR DATA SCIENTIST'
		WHEN exp > 2 AND EXP  <= 5 THEN 'ASSOCIATE DATA SCIENTIST'
		WHEN exp > 5 AND EXP  <= 10 THEN 'SENIOR DATA SCIENTIST'
		WHEN exp > 10 AND EXP  <= 12 THEN 'LEAD DATA SCIENTIST'
		WHEN exp > 12 AND EXP  <= 16 THEN 'MANAGER'
	END
```

_lets remove this __designation__ column before moving on_  


```sql
ALTER TABLE emp_record_table
	DROP COLUMN designation
```

### 15. Create an index to improve the cost and performance of the query to find the employee whose FIRST_NAME is ‘Eric’ in the employee table after checking the execution plan.  

- index helps in speeding up queries, its performance is quite observable in large datasets.  
- index actually creates a backend table that would keep a ~binary record of these values.  
- text was not compatible with indexing so we need to change to __VARCHAR__  


```sql
ALTER TABLE emp_record_table
	MODIFY COLUMN first_name VARCHAR(20)
```



```sql
CREATE INDEX ix_firstname ON emp_record_table(FIRST_NAME)
```

- This table is too short to tell the difference in query time  


```sql
SELECT * FROM emp_record_table WHERE first_name = "eric"
```


<div class="knitsql-table">


Table: 1 records

|emp_id |first_name |LAST_NAME |GENDER |ROLE                |DEPT    | EXP|COUNTRY |CONTINENT     | SALARY| EMP_RATING|MANAGER_ID |proj_id |designation |
|:------|:----------|:---------|:------|:-------------------|:-------|---:|:-------|:-------------|------:|----------:|:----------|:-------|:-----------|
|E005   |Eric       |Hoffman   |M      |LEAD DATA SCIENTIST |FINANCE |  11|USA     |NORTH AMERICA |   8500|          3|E103       |P105    |NA          |

</div>

### 16. Write a query to calculate the bonus for all the employees, based on their ratings and salaries (Use the formula: 5% of salary * employee rating).


```sql
SELECT * , salary*0.05*emp_rating as bonus
	FROM emp_record_table
```


<div class="knitsql-table">


Table: Displaying records 1 - 10

|emp_id |first_name |LAST_NAME |GENDER |ROLE                  |DEPT       | EXP|COUNTRY |CONTINENT     | SALARY| EMP_RATING|MANAGER_ID |proj_id |designation | bonus|
|:------|:----------|:---------|:------|:---------------------|:----------|---:|:-------|:-------------|------:|----------:|:----------|:-------|:-----------|-----:|
|E001   |Arthur     |Black     |M      |PRESIDENT             |ALL        |  20|USA     |NORTH AMERICA |  16500|          5|NA         |NA      |NA          |  4125|
|E005   |Eric       |Hoffman   |M      |LEAD DATA SCIENTIST   |FINANCE    |  11|USA     |NORTH AMERICA |   8500|          3|E103       |P105    |NA          |  1275|
|E010   |William    |Butler    |M      |LEAD DATA SCIENTIST   |AUTOMOTIVE |  12|FRANCE  |EUROPE        |   9000|          2|E428       |P204    |NA          |   900|
|E052   |Dianna     |Wilson    |F      |SENIOR DATA SCIENTIST |HEALTHCARE |   6|CANADA  |NORTH AMERICA |   5500|          5|E083       |P103    |NA          |  1375|
|E057   |Dorothy    |Wilson    |F      |SENIOR DATA SCIENTIST |HEALTHCARE |   9|USA     |NORTH AMERICA |   7700|          1|E083       |P302    |NA          |   385|
|E083   |Patrick    |Voltz     |M      |MANAGER               |HEALTHCARE |  15|USA     |NORTH AMERICA |   9500|          5|E001       |NA      |NA          |  2375|
|E103   |Emily      |Grove     |F      |MANAGER               |FINANCE    |  14|CANADA  |NORTH AMERICA |  10500|          4|E001       |NA      |NA          |  2100|
|E204   |Karene     |Nowak     |F      |SENIOR DATA SCIENTIST |AUTOMOTIVE |   8|GERMANY |EUROPE        |   7500|          5|E428       |P204    |NA          |  1875|
|E245   |Nian       |Zhen      |M      |SENIOR DATA SCIENTIST |RETAIL     |   6|CHINA   |ASIA          |   6500|          2|E583       |P109    |NA          |   650|
|E260   |Roy        |Collins   |M      |SENIOR DATA SCIENTIST |RETAIL     |   7|INDIA   |ASIA          |   7000|          3|E583       |NA      |NA          |  1050|

</div>

### 17. Write a query to calculate the average salary distribution based on the continent and country. Take data from the employee record table.


```sql
SELECT CONTINENT, COUNTRY, AVG(SALARY) AS average_salary
FROM 
    emp_record_table
GROUP BY 
    continent, country;
```


<div class="knitsql-table">


Table: 7 records

|CONTINENT     |COUNTRY  | average_salary|
|:-------------|:--------|--------------:|
|NORTH AMERICA |USA      |       9440.000|
|EUROPE        |FRANCE   |       9000.000|
|NORTH AMERICA |CANADA   |       7000.000|
|EUROPE        |GERMANY  |       7600.000|
|ASIA          |CHINA    |       6500.000|
|ASIA          |INDIA    |       6166.667|
|SOUTH AMERICA |COLOMBIA |       5600.000|

</div>
