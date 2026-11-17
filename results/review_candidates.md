# Manual review candidates

Sampled with a fixed seed from Spider dev (test-A). SFT = row 2 best-on-val, GRPO = row 3 best-on-val.


## GRPO right, SFT wrong (102 total, 5 sampled)

**spider_dev_545** (student_transcripts_tracking): What are all the course names of the courses which ever have students enrolled in?

```sql
-- gold
SELECT DISTINCT T1.course_name FROM Courses AS T1 JOIN Student_Enrolment_Courses AS T2 ON T1.course_id  =  T2.course_id
-- SFT (no_such_column)
SELECT T2.section_name FROM Student_Enrolment AS T1 JOIN Sections AS T2 ON T1.course_id  =  T2.course_id
-- GRPO (ok)
SELECT course_name FROM Courses WHERE course_id IN (SELECT course_id FROM Student_Enrolment_Courses)
```

**spider_dev_942** (dog_kennels): Which professionals have done at least two types of treatments? List the professional id and cell phone.

```sql
-- gold
SELECT T1.professional_id ,  T1.cell_number FROM Professionals AS T1 JOIN Treatments AS T2 ON T1.professional_id  =  T2.professional_id GROUP BY T1.professional_id HAVING count(*)  >=  2
-- SFT (wrong result)
SELECT T1.first_name ,  T1.cell_number FROM professionals AS T1 JOIN treatments AS T2 ON T1.professional_id  =  T2.professional_id GROUP BY T1.professional_id HAVING COUNT (DISTINCT T2.treatment_type_code)  >=  2
-- GRPO (ok)
SELECT p.professional_id, p.cell_number FROM Professionals p JOIN Treatments t ON p.professional_id = t.professional_id GROUP BY p.professional_id HAVING count(*) >= 2
```

**spider_dev_566** (student_transcripts_tracking): What is the earliest date of a transcript release, and what details can you tell me?

```sql
-- gold
SELECT transcript_date ,  other_details FROM Transcripts ORDER BY transcript_date ASC LIMIT 1
-- SFT (no_such_table)
SELECT min(transcript_date) ,  other_details FROM TRANSCRIPT
-- GRPO (ok)
SELECT min(transcript_date), other_details FROM Transcripts
```

**spider_dev_40** (concert_singer): What is the name and country of origin of every singer who has a song with the word 'Hey' in its title?

```sql
-- gold
SELECT name ,  country FROM singer WHERE song_name LIKE '%Hey%'
-- SFT (no_such_table)
SELECT T1.name ,  T1.country FROM singer AS T1 JOIN song AS T2 ON T1.singer_id  =  T2.singer_id WHERE T2.song_name LIKE '%Hey%'
-- GRPO (ok)
SELECT Name, Country FROM singer WHERE Singer_ID IN (SELECT Singer_ID FROM singer WHERE Song_Name LIKE '%Hey%')
```

**spider_dev_356** (cre_Doc_Template_Mgt): Return the ids corresponding to templates with the description 'Presentation'.

```sql
-- gold
SELECT T2.template_id FROM Ref_template_types AS T1 JOIN Templates AS T2 ON T1.template_type_code  = T2.template_type_code WHERE T1.template_type_description  =  "Presentation"
-- SFT (no_such_column)
SELECT template_id_code FROM Ref_Template_Types WHERE template_type_description  =  "Presentation"
-- GRPO (ok)
SELECT Template_ID FROM Templates WHERE Template_Type_Code IN (SELECT Template_Type_Code FROM Ref_Template_Types WHERE Template_Type_Description = 'Presentation')
```


## GRPO wrong, SFT right (95 total, 5 sampled)

**spider_dev_749** (world_1): Give the names of countries with English and French as official languages.

```sql
-- gold
SELECT T1.Name FROM country AS T1 JOIN countrylanguage AS T2 ON T1.Code  =  T2.CountryCode WHERE T2.Language  =  "English" AND T2.IsOfficial  =  "T" INTERSECT SELECT T1.Name FROM country AS T1 JOIN countrylanguage AS T2 ON T1.Code  =  T2.CountryCode WHERE T2.Language  =  "French" AND T2.IsOfficial  =  "T"
-- SFT (ok)
SELECT T1.Name FROM country AS T1 JOIN countrylanguage AS T2 ON T1.Code  =  T2.CountryCode WHERE T2.IsOfficial  =  'T' AND T2.Language  =  'English' INTERSECT SELECT T1.Name FROM country AS T1 JOIN countrylanguage AS T2 ON T1.Code  =  T2.CountryCode WHERE T2.IsOfficial  =  'T' AND T2.Language  =  'French'
-- GRPO (wrong result)
SELECT Name FROM country WHERE Code IN (SELECT CountryCode FROM countrylanguage WHERE Language = 'English') INTERSECT SELECT Name FROM country WHERE Code IN (SELECT CountryCode FROM countrylanguage WHERE Language = 'French')
```

**spider_dev_734** (world_1): How many type of governments are in Africa?

```sql
-- gold
SELECT count(DISTINCT GovernmentForm) FROM country WHERE Continent  =  "Africa"
-- SFT (ok)
SELECT count(DISTINCT governmentform) FROM country WHERE continent  =  'Africa'
-- GRPO (wrong result)
SELECT count(DISTINCT GovernmentForm) FROM country WHERE Code IN (SELECT Code FROM country WHERE Region = 'Africa')
```

**spider_dev_635** (tvshow): which countries' tv channels are not playing any cartoon written by Todd Casey?

```sql
-- gold
SELECT country FROM TV_Channel EXCEPT SELECT T1.country FROM TV_Channel AS T1 JOIN cartoon AS T2 ON T1.id = T2.Channel WHERE T2.written_by  =  'Todd Casey'
-- SFT (ok)
SELECT country FROM TV_channel EXCEPT SELECT T1.country FROM TV_channel AS T1 JOIN cartoon AS T2 ON T1.id  =  T2.channel WHERE T2.written_by  =  "Todd Casey"
-- GRPO (wrong result)
SELECT Country FROM TV_Channel WHERE id NOT IN (SELECT Channel FROM Cartoon WHERE Written_by = 'Todd Casey')
```

**spider_dev_496** (battle_death): What are the death and injury situations caused by the ship with tonnage 't'?

```sql
-- gold
SELECT T1.killed ,  T1.injured FROM death AS T1 JOIN ship AS t2 ON T1.caused_by_ship_id  =  T2.id WHERE T2.tonnage  =  't'
-- SFT (ok)
SELECT T1.killed ,  T1.injured FROM death AS T1 JOIN ship AS T2 ON T1.caused_by_ship_id  =  T2.id WHERE T2.tonnage  =  't'
-- GRPO (wrong result)
SELECT note, killed, injured FROM death WHERE caused_by_ship_id IN (SELECT id FROM ship WHERE tonnage = 't')
```

**spider_dev_729** (world_1): What is the total population of Gelderland district?

```sql
-- gold
SELECT sum(Population) FROM city WHERE District  =  "Gelderland"
-- SFT (ok)
SELECT sum(population) FROM city WHERE district  =  'Gelderland'
-- GRPO (wrong result)
SELECT Population FROM city WHERE District = 'Gelderland'
```


## Both wrong (213 total, 5 sampled)

**spider_dev_454** (wta_1): What is the first name and country code of the oldest player?

```sql
-- gold
SELECT first_name ,  country_code FROM players ORDER BY birth_date LIMIT 1
-- SFT (wrong result)
SELECT first_name ,  country_code FROM players ORDER BY birth_date DESC LIMIT 1
-- GRPO (wrong result)
SELECT first_name, country_code FROM players ORDER BY birth_date DESC LIMIT 1
```

**spider_dev_740** (world_1): What is name of the country that speaks the largest number of languages?

```sql
-- gold
SELECT T1.Name FROM country AS T1 JOIN countrylanguage AS T2 ON T1.Code  =  T2.CountryCode GROUP BY T1.Name ORDER BY COUNT(*) DESC LIMIT 1
-- SFT (wrong result)
SELECT T1.Name FROM country AS T1 JOIN countrylanguage AS T2 ON T1.Code  =  T2.CountryCode GROUP BY T2.CountryCode ORDER BY count(*) DESC LIMIT 1
-- GRPO (wrong result)
SELECT Name FROM country WHERE Code IN (SELECT CountryCode FROM countrylanguage GROUP BY CountryCode ORDER BY count(*) DESC LIMIT 1)
```

**spider_dev_168** (car_1): Among the cars that do not have the minimum horsepower , what are the make ids and names of all those with less than 4 cylinders ?

```sql
-- gold
select t2.makeid ,  t2.make from cars_data as t1 join car_names as t2 on t1.id  =  t2.makeid where t1.horsepower  >  (select min(horsepower) from cars_data) and t1.cylinders  <  4;
-- SFT (no_such_column)
SELECT id ,  make ,  model FROM car_names WHERE cylinders  <  4 AND id != (SELECT id FROM cars_data ORDER BY horsepower LIMIT 1)
-- GRPO (no_such_column)
SELECT c.MakeId, c.Make FROM car_names c WHERE c.MakeId NOT IN (SELECT Id FROM cars_data WHERE Horsepower = (SELECT MIN(Horsepower) FROM cars_data)) AND c.Cylinders < 4
```

**spider_dev_628** (tvshow): How many cartoons did each director create?

```sql
-- gold
SELECT count(*) ,  Directed_by FROM cartoon GROUP BY Directed_by
-- SFT (wrong result)
SELECT directed_by ,  count(*) FROM Cartoon GROUP BY directed_by
-- GRPO (wrong result)
SELECT Directed_by, count(*) FROM Cartoon GROUP BY Directed_by
```

**spider_dev_115** (car_1): How many car models are produced in the usa?

```sql
-- gold
SELECT count(*) FROM MODEL_LIST AS T1 JOIN CAR_MAKERS AS T2 ON T1.Maker  =  T2.Id JOIN COUNTRIES AS T3 ON T2.Country  =  T3.CountryId WHERE T3.CountryName  =  'usa';
-- SFT (wrong result)
SELECT count(*) FROM car_makers AS T1 JOIN model_list AS T2 ON T1.Id  =  T2.Maker WHERE T1.Country  =  "USA"
-- GRPO (wrong result)
SELECT count(*) FROM model_list WHERE Maker IN (SELECT Id FROM car_makers WHERE Country = 'USA')
```


## Both right, different SQL (check for coincidental matches) (438 total, 5 sampled)

**spider_dev_323** (cre_Doc_Template_Mgt): What are the ids of templates with template type code PP or PPT?

```sql
-- gold
SELECT template_id FROM Templates WHERE template_type_code  =  "PP" OR template_type_code  =  "PPT"
-- SFT (ok)
SELECT template_id FROM Templates WHERE template_type_code  =  "PP" OR template_type_code  =  "PPT"
-- GRPO (ok)
SELECT Template_ID FROM Templates WHERE Template_Type_Code = 'PP' OR Template_Type_Code = 'PPT'
```

**spider_dev_185** (flight_2): List the airport code and name in the city of Anthony.

```sql
-- gold
SELECT AirportCode ,  AirportName FROM AIRPORTS WHERE city  =  "Anthony"
-- SFT (ok)
SELECT AirportCode ,  AirportName FROM airports WHERE City  =  "Anthony"
-- GRPO (ok)
SELECT AirportCode, AirportName FROM airports WHERE City = 'Anthony'
```

**spider_dev_917** (network_1): What is the lowest grade of students who do not have any friends?

```sql
-- gold
SELECT min(grade) FROM Highschooler WHERE id NOT IN (SELECT T1.student_id FROM Friend AS T1 JOIN Highschooler AS T2 ON T1.student_id  =  T2.id)
-- SFT (ok)
SELECT min(grade) FROM Highschooler WHERE id NOT IN (SELECT student_id FROM Friend)
-- GRPO (ok)
SELECT min(grade) FROM Highschooler WHERE ID NOT IN (SELECT student_id FROM Friend)
```

**spider_dev_118** (car_1): What is the average miles per gallon of all the cards with 4 cylinders?

```sql
-- gold
SELECT avg(mpg) FROM CARS_DATA WHERE Cylinders  =  4;
-- SFT (ok)
SELECT avg(MPG) FROM cars_data WHERE Cylinders  =  4
-- GRPO (ok)
SELECT AVG(MPG) FROM cars_data WHERE Cylinders = 4
```

**spider_dev_701** (voter_1): Return the names of the contestants whose names contain the substring 'Al' .

```sql
-- gold
select contestant_name from contestants where contestant_name like "%al%"
-- SFT (ok)
SELECT contestant_name FROM CONTESTANTS WHERE contestant_name LIKE '%Al%'
-- GRPO (ok)
SELECT contestant_name FROM CONTESTANTS WHERE contestant_name LIKE '%Al%'
```
