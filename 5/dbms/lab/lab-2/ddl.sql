CREATE DATABASE university_fest_db;
USE university_fest_db;

CREATE TABLE teams (
    team_id INT PRIMARY KEY,
    team_name VARCHAR(100) NOT NULL,
    num_members INT,
    team_type ENUM('MNG', 'ORG') DEFAULT 'ORG',
    fest_id INT
);

SHOW CREATE TABLE teams;

CREATE TABLE fest (
    fest_id INT PRIMARY KEY,
    fest_name VARCHAR(100) UNIQUE NOT NULL,
    year YEAR NOT NULL,
    head_team_id INT NOT NULL,
    FOREIGN KEY (head_team_id) REFERENCES teams(team_id)
);

SHOW CREATE TABLE fest;

ALTER TABLE teams ADD FOREIGN KEY (fest_id) REFERENCES fest(fest_id) ON DELETE CASCADE;

CREATE TABLE members (
    mem_id INT PRIMARY KEY,
    mem_name VARCHAR(100) NOT NULL,
    dob DATE NOT NULL,
    age INT NOT NULL,
    super_mem_id INT,
    team_id INT NOT NULL,
    FOREIGN KEY (super_mem_id) REFERENCES members(mem_id),
    FOREIGN KEY (team_id) REFERENCES teams(team_id) ON DELETE CASCADE
);

SHOW CREATE TABLE members;

CREATE TABLE events (
    event_id INT PRIMARY KEY,
    event_name VARCHAR(100) NOT NULL,
    building VARCHAR(50),
    floor INT,
    room_no VARCHAR(20),
    price INT CHECK(price <= 1500) DEFAULT 0,
    team_id INT NOT NULL,
    FOREIGN KEY (team_id) REFERENCES teams(team_id) ON DELETE CASCADE
);

SHOW CREATE TABLE events;

CREATE TABLE event_conduction (
    event_id INT,
    date_of_conduction DATE,
    PRIMARY KEY (event_id, date_of_conduction),
    FOREIGN KEY (event_id) REFERENCES events(event_id) ON DELETE CASCADE
);

SHOW CREATE TABLE event_conduction;

CREATE TABLE participants (
    srn VARCHAR(20) PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    department VARCHAR(100),
    semester INT,
    gender ENUM('M', 'F', 'O')
);

SHOW CREATE TABLE participants;

CREATE TABLE visitors (
    visitor_id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    age INT,
    gender ENUM('M', 'F', 'O'),
    participant_srn VARCHAR(20) NOT NULL,
    FOREIGN KEY (participant_srn) REFERENCES participants(srn)
);

SHOW CREATE TABLE visitors;

CREATE TABLE registration (
    event_id INT,
    srn VARCHAR(20),
    registration_number INT NOT NULL,
    PRIMARY KEY (event_id, srn),
    FOREIGN KEY (event_id) REFERENCES events(event_id) ON DELETE CASCADE,
    FOREIGN KEY (srn) REFERENCES participants(srn)
);

SHOW CREATE TABLE registration;

CREATE TABLE stalls (
    stall_id INT PRIMARY KEY,
    stall_name VARCHAR(100) UNIQUE NOT NULL,
    fest_id INT NOT NULL,
    FOREIGN KEY (fest_id) REFERENCES fest(fest_id) ON DELETE CASCADE
);

SHOW CREATE TABLE stalls;

CREATE TABLE items (
    item_name VARCHAR(100) PRIMARY KEY,
    type ENUM('Veg', 'Non-veg') NOT NULL
);

SHOW CREATE TABLE items;

CREATE TABLE stall_items (
    stall_id INT,
    item_name VARCHAR(100),
    price_per_unit DECIMAL(8,2) NOT NULL,
    total_quantity INT DEFAULT 0,
    PRIMARY KEY (stall_id, item_name),
    FOREIGN KEY (stall_id) REFERENCES stalls(stall_id) ON DELETE CASCADE,
    FOREIGN KEY (item_name) REFERENCES items(item_name)
);

SHOW CREATE TABLE stall_items;

CREATE TABLE purchases (
    purchase_id INT AUTO_INCREMENT PRIMARY KEY,
    srn VARCHAR(20) NOT NULL,
    stall_id INT NOT NULL,
    item_name VARCHAR(100) NOT NULL,
    purchase_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    quantity INT NOT NULL,
    total_amount DECIMAL(10,2),
    FOREIGN KEY (srn) REFERENCES participants(srn),
    FOREIGN KEY (stall_id, item_name) REFERENCES stall_items(stall_id, item_name)
);

SHOW CREATE TABLE purchases;


SHOW CREATE TABLE participants;
ALTER TABLE participants MODIFY COLUMN gender ENUM('M', 'F', 'O') NOT NULL AFTER name;
SHOW CREATE TABLE participants;

SHOW CREATE TABLE stall_items;
ALTER TABLE stall_items MODIFY COLUMN price_per_unit DECIMAL(6, 2) NOT NULL DEFAULT 50;
SHOW CREATE TABLE stall_items;

SHOW CREATE TABLE stall_items;
ALTER TABLE stall_items ADD CONSTRAINT chk_max_stock CHECK (Total_quantity <= 150);
SHOW CREATE TABLE stall_items;

SHOW CREATE TABLE event_conduction;
RENAME TABLE event_conduction TO event_schedule;
SHOW CREATE TABLE event_schedule;

SHOW CREATE TABLE event_schedule;
ALTER TABLE event_schedule MODIFY COLUMN date_of_conduction DATE NOT NULL FIRST;
SHOW CREATE TABLE event_schedule;