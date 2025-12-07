show databases;
use Fest_Database;

show tables;

select * from event;

INSERT INTO event
VALUES ('E33', 'AI Hackathon', 'Seminar Hall', '2', 205, 900.00, 'T4');

select * from event;

select * from stall_items;

update stall_items set total_quantity=25 where item_name='Mushroom Risotto';

select * from stall_items;

select * from registration;

delete from registration where event_id = 'E1' and SRN like 'P100%';

select * from registration;

select * from purchased;

insert into purchased values ('P1017', 'S6', 'Fish Tacos', '2025-07-10
14:00:00', 3);

select * from purchased;

select * from registration;

select srn from registration where event_id = 'E2' or event_id = 'E5'
and srn not in
(select srn from registration where event_id = 'E2'
intersect
select srn from registration where event_id = 'E5');

show tables;

select * from visitor;

select p.name as participant_name, 
group_concat(v.name separator ',') as visitor_name,
count(v.name) as visitor_count
from participant p
left join visitor v on p.SRN = v.SRN 
group by participant_name
order by visitor_count DESC;

show tables;

select r.event_id from registration r 
join participant p on r.SRN = p.SRN
group by r.event_id
having sum(p.gender = 'Male') = sum(p.gender = 'Female');

select event_name, event_conduction.date_of_conduction,
case
when year(event_conduction.date_of_conduction) > 2047 then 1
else 0
end as after_golden_jubilee_year
 from event
join event_conduction on event.event_id = event_conduction.event_id;