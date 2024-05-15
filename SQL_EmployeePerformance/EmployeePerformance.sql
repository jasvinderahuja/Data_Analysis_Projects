CREATE SCHEMA employee

use employee

SELECT * FROM proj_table

ALTER TABLE proj_table
	MODIFY COLUMN project_id VARCHAR(10)

ALTER TABLE proj_table
	ADD PRIMARY KEY(project_id)
    
---
SELECT * FROM emp_record_table

ALTER TABLE emp_record_table
	MODIFY COLUMN emp_id VARCHAR(10)
    
ALTER TABLE emp_record_table
	ADD PRIMARY KEY(emp_id)

DESCRIBE emp_record_table
---

-- some proj_id in the employee table are NA
-- FOREIGN KEY needs to match the tables.

-- Error: 1175
-- Notes from video-6 Amit Goyal ~41 minutes
-- go to menu "MySQLWorkbench" > "Settings" > "SQL Editor" > uncheck "Safe Updates"
-- on PC I think it is under Edit

UPDATE emp_record_table
	SET proj_id=NULL
    WHERE proj_id='NA'

ALTER TABLE emp_record_table
	MODIFY COLUMN proj_id VARCHAR(10)
    
ALTER TABLE emp_record_table
	ADD CONSTRAINT fk_proj
    FOREIGN KEY(proj_id) REFERENCES proj_table(PROJECT_ID)
    
---

SELECT * FROM data_science_team

DESCRIBE data_science_team

ALTER TABLE data_science_team
	MODIFY COLUMN emp_id VARCHAR(10)
    
ALTER TABLE data_science_team
	ADD CONSTRAINT fk_emp_record_table_emp_id
	FOREIGN KEY(emp_id) REFERENCES emp_record_table(emp_id)
    
-- create EED

-- goto "database" > "Reverse Engineer" and follow prompts

--
SELECT emp_id, first_name, last_name, gender, dept
	FROM emp_record_table
    WHERE emp_rating < 2 OR emp_rating BETWEEN 2 AND 4 OR emp_rating > 4

-- 5 --

SELECT CONCAT(first_name, ",", last_name) AS NAME FROM emp_record_table
	WHERE dept='FINANCE'

-- 6 --

SELECT * FROM emp_record_table
	WHERE emp_id IN (SELECT DISTINCT manager_id from emp_record_table)
    
-- 7 --

SELECT * FROM emp_record_table
	WHERE dept = 'HEALTHCARE'
UNION
SELECT * FROM emp_record_table
	WHERE dept = 'FINANCE'

-- 8 --

-- Write a query to list down employee details such as EMP_ID, FIRST_NAME, LAST_NAME, ROLE, DEPARTMENT, and EMP_RATING grouped by dept. 
-- Also include the respective employee rating along with the max emp rating for the department.

SELECT emp_id, first_name, last_name, role, dept, MAX(emp_rating) AS max_emp_rating
	FROM emp_record_table
-- this GROUP BY is used for max_emp_rating
	GROUP BY emp_id, first_name, last_name, role, dept
    

-- 9 --

SELECT ROLE, MIN(salary), MAX(salary) 
	FROM emp_record_table
    GROUP BY ROLE


-- 10 --

SELECT *, RANK() OVER (ORDER by exp DESC)
	FROM emp_record_table
    
SELECT *: This selects all columns from the emp_record_table.

-- RANK() OVER (ORDER by exp DESC): This is a window function that calculates the rank of each record based on the exp column in descending order. 
-- The ORDER BY exp DESC specifies that the records should be ranked based on the exp column in descending ord

-- 11 -- 
-- for complex queries to not have to write them again and again
DROP VIEW IF EXISTS vw_country

CREATE VIEW vw_country
	AS
    SELECT first_name, last_name, country FROM emp_record_table
    WHERE salary > 6000


-- 12 -- 
SELECT * FROM
(
SELECT * FROM emp_record_table
	WHERE exp > 10
) AS T
    
-- 13 --

delimiter //
CREATE PROCEDURE sp_exp()
	BEGIN
		SELECT * FROM emp_record_table
        WHERE exp > 3;
	END
delimiter //

-- Here, // is used as the delimiter. It's changed back to ; after the stored procedure definition. 
-- This ensures that the SQL interpreter doesn't interpret the semicolons within the stored procedure as statement terminators until the entire procedure is defined.

-- 14 --

SELECT * ,
	CASE WHEN exp <= 2 THEN 'JUNIOR DATA SCIENTIST'
    WHEN exp > 2 AND EXP  <= 5 THEN 'ASSOCIATE DATA SCIENTIST'
    WHEN exp > 5 AND EXP  <= 10 THEN 'SENIOR DATA SCIENTIST'
    WHEN exp > 10 AND EXP  <= 12 THEN 'LEAD DATA SCIENTIST'
    WHEN exp > 12 AND EXP  <= 16 THEN 'MANAGER'
    END AS designation
FROM emp_record_table

-- to add a column like this 

ALTER TABLE emp_record_table
	ADD designation VARCHAR(50)

UPDATE emp_record_table
	SET designation=CASE WHEN exp <= 2 THEN 'JUNIOR DATA SCIENTIST'
		WHEN exp > 2 AND EXP  <= 5 THEN 'ASSOCIATE DATA SCIENTIST'
		WHEN exp > 5 AND EXP  <= 10 THEN 'SENIOR DATA SCIENTIST'
		WHEN exp > 10 AND EXP  <= 12 THEN 'LEAD DATA SCIENTIST'
		WHEN exp > 12 AND EXP  <= 16 THEN 'MANAGER'
	END

-- lets remove this column before moving on
ALTER TABLE emp_record_table
	DROP COLUMN designation

-- 15 -- 
-- index helps in speeding up queries, it is quite obserable in large datasets.
-- index actually creates a backend table that would keep a ~binary record of these values.
-- text was not compatible with indexing
ALTER TABLE emp_record_table
	MODIFY COLUMN first_name VARCHAR(20)

CREATE INDEX ix_firstname ON emp_record_table(FIRST_NAME)

-- 16 --
SELECT * , salary*0.05*emp_rating as bonus
	FROM emp_record_table

-- 17 -- 2h20m

SELECT 

-- PARTITION -- helps in rank by deparment
SELECT * , rank() over (ORDER BY salary desc)
	FROM emp_record_table
    
-- vs

SELECT * , rank() over (PARTITION BY dept ORDER BY salary desc)
	FROM emp_record_table

