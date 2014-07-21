.mode column
.headers on

CREATE TABLE DF1 (
    index_ integer,
    date_ text,
    day_week text,
    followers integer,
    id integer,
    retweet_count integer,
    screen_name text,
    text text,
    time text,
    user_id integer
);

.separator ','
.import ./CNBC.csv DF1

