-- Initialize database schema for Emotion Classification System
-- This file is automatically executed when the MySQL container starts

USE emotion_db;

-- Ensure database is using UTF-8
ALTER DATABASE emotion_db CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

-- The tables will be created by SQLAlchemy's init_db() function
-- This file is here for any additional initialization if needed

-- Create a read-only user for analytics (optional)
-- CREATE USER IF NOT EXISTS 'emotion_readonly'@'%' IDENTIFIED BY 'readonly_password';
-- GRANT SELECT ON emotion_db.* TO 'emotion_readonly'@'%';
-- FLUSH PRIVILEGES;
