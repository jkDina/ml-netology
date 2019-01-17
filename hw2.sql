CREATE TABLE IF NOT EXISTS Students (
	id int,
	name VARCHAR (30),
	birthday date
);

CREATE TABLE IF NOT EXISTS Professors (
	id int,
	name VARCHAR (30)
);

CREATE TABLE IF NOT EXISTS Courses (
	id int,
	name VARCHAR (30)
);

CREATE TABLE IF NOT EXISTS Professors_Courses (
	professor_id integer,
	course_id integer
);

CREATE TABLE IF NOT EXISTS Courses_Students (
	course_id integer,
	student_id integer
);

SELECT * FROM Courses_Students;

--SELECT course_id, COUNT(student_id) AS student_count FROM Courses_Students GROUP BY course_id;
--SELECT COUNT(id) AS st_count FROM Students;
--SELECT COUNT(id) AS cr_count FROM Courses;
--SELECT s.cr_count FROM (SELECT COUNT(id) AS cr_count FROM Courses) AS s(cr_count);
--SELECT s.cr_count FROM (SELECT COUNT(id) AS cr_count FROM Courses);
--SELECT Courses.id AS cr_count, Students.id AS st_count FROM Courses CROSS JOIN Students;
--SELECT SUM(course_size) / COUNT(course_id) * 0.5 AS threshold FROM (SELECT COUNT(student_id) AS course_size, course_id FROM Courses_Students GROUP BY course_id) AS t;
--SELECT * FROM Professors WHERE 
SELECT DISTINCT professor_id FROM 
Professors_Courses AS pc 
LEFT JOIN 
(SELECT course_id, COUNT(student_id) AS course_size FROM Courses_Students GROUP BY course_id) 
AS t ON pc.course_id = t.course_id CROSS JOIN
(SELECT SUM(course_size) / COUNT(course_id) * 0.5 AS threshold FROM 
(SELECT COUNT(student_id) AS course_size, course_id FROM Courses_Students GROUP BY course_id) AS t) AS th
WHERE course_size < threshold;
