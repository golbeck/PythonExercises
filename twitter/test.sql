.mode column
.headers on

CREATE TABLE DF1 (
    index_ integer,
    created_at text,
    followers integer,
    screen_name text,
    text text
);

.separator ','
.import ./test.csv DF1

