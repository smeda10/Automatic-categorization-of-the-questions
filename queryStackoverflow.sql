SELECT * FROM posts WHERE (Score + CommentCount >= 8) AND Id < 280000);
SELECT * FROM posts WHERE (Score + CommentCount >= 8) AND (Id >= 280000 AND Id < 500000);
SELECT * FROM posts WHERE (Score + CommentCount >= 8) AND (Id >= 500000 AND Id < 750000);

