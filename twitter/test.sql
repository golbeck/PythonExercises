.mode column
.headers on

CREATE TABLE DF1 (
    index_ integer,
    date_ text,
    day_week text,
    followers integer,
    id integer,
    screen_name text,
    text text,
    time text
);

.separator ','
.import ./test.csv DF1

