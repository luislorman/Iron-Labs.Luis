use sakila;

## Lab | SQL Intro
#1. Review the tables in the database.
show columns from actor; # this si info() in Python
#2. Explore tables by selecting all columns from each table or using the in built review features for your client.
select * from film;

#3. Select one column from a table. Get film titles.
select  title from film;

#4. Select one column from a table and alias it. Get unique list of film languages under the alias `language`. Note that we are not asking you to obtain the language per each film, but this is a good time to think about how you might get that information in the future.
select title as Language, original_language_id  from film
where title LIKE '%language%';

#5.
#* 5.1 Find out how many stores does the company have?
select count(store_id) from store;  # 2 stores

#* 5.2 Find out how many employees staff does the company have? 
select count(staff_id) from staff; # 2 employees

#* 5.3 Return a list of employee first names only?
select first_name from staff; # Mike and Jon

#############################################################################################
# Introduction

#In this lab, you will be using the Bank database.
#Here, we will practice selecting and projecting data. You can finish all questions with only those clauses:
#- `SELECT`
#- `SELECT DISTINCT`
#- `FROM`
#- `WHERE`
#- `ORDER BY`
#- `LIMIT`

## Instructions

#Assume that any `_id` columns are incremental, meaning that higher ids always occur after lower ids. 
#For example, a client with a higher `client_id` joined the bank after a client with a lower `client_id`.

### Query 1

#Get the `id` values of the first 5 clients from `district_id` with a value equals to 1.
select client_id from bank.client where district_id = 1;

### Query 2
#In the `client` table, get an `id` value of the last client where the `district_id` equals to 72.
select max(client_id)  from bank.client where district_id = 72;

### Query 3
#Get the 3 lowest amounts in the `loan` table.
select amount from bank.loan order by amount asc limit 3;

### Query 4
#What are the possible values for `status`, ordered alphabetically in ascending order in the `loan` table?
 select status from bank.loan order by status asc;

### Query 5
#What is the `loan_id` of the highest payment received in the `loan` table?
#2 ways -- WAY A
select max(payments) from bank.loan;      
select loan_id from bank.loan where  payments=9910;
#WAY B
select loan_id from bank.loan order by payments desc;

### Query 6
#What is the loan `amount` of the lowest 5 `account_id`s in the `loan` table? Show the `account_id` and the corresponding `amount`
select account_id, amount from bank.loan order by account_id asc limit 5;

### Query 7
#What are the top 5 `account_id`s with the lowest loan `amount` that have a loan `duration` of 60 in the `loan` table?
select account_id from bank.loan where duration= 60 order by amount asc limit 5;

### Query 8
#What are the unique values of `k_symbol` in the `order` table?
select distinct (k_symbol) from bank.order;

### Query 9
#In the `order` table, what are the `order_id`s of the client with the `account_id` 34?
#Note: There shouldn't be a table name `order`, since `order` is reserved from the `ORDER BY` clause. 
#You have to use backticks to escape the `order` table name.
select order_id from bank.order where account_id = 34;

### Query 10
#In the `order` table, which `account_id`s were responsible for orders between `order_id` 29540 and `order_id` 29560 (inclusive)?
select account_id from bank.order where order_id > 29540 and order_id < 29560;

### Query 11
#In the `order` table, what are the individual amounts that were sent to (`account_to`) id 30067122?
select amount from bank.order where account_to=30067122;

### Query 12
#In the `trans` table, show the `trans_id`, `date`, `type` and `amount` of the 10 first transactions 
#from `account_id` 793 in chronological order, from newest to oldest.
select trans_id, date, type, amount from bank.trans where account_id=793  order by date desc limit 10;

# Optional
### Query 13
#In the `client` table, of all districts with a `district_id` lower than 10, how many clients are from each `district_id`? 
#Show the results sorted by the `district_id` in ascending order.

# Optional
### Query 13
#In the `client` table, of all districts with a `district_id` lower than 10, how many clients are from each `district_id`? 
#Show the results sorted by the `district_id` in ascending order.
select count(client_id), district_id from bank.client
where district_id < 10 
group by district_id
order by district_id asc;

### Query 14
#In the `card` table, how many cards exist for each `type`? Rank the result starting with the most frequent `type`.
select count(card_id), type from bank.card
group by type
order by card_id desc;

### Query 15
#Using the `loan` table, print the top 10 `account_id`s based on the sum of all of their loan amounts.
select account_id, amount from bank.loan group by amount order by amount desc limit 10;

### Query 16
#In the `loan` table, retrieve the number of loans issued for each day, before (excl) 930907, ordered by date in descending order.
select count(loan_id), date from bank.loan where date<930907  group by date order by date desc;

### Query 17
#In the `loan` table, for each day in December 1997, count the number of loans issued for each unique loan duration, ordered by date and duration, both in ascending order. 
#You can ignore days without any loans in your output.

select date from bank.loan;
#WAY A
select count(loan_id), date,duration from bank.loan where date >=  971200 and date <=971231 group by duration order by date and duration asc;
#WAY B
select count(loan_id), date,duration from bank.loan where date between 971200 and 971231 group by duration order by date and duration asc;

### Query 18
#In the `trans` table, for `account_id` 396, sum the amount of transactions for each type (`VYDAJ` = Outgoing, `PRIJEM` = Incoming). 
#Your output should have the `account_id`, the `type` and the sum of amount, named as `total_amount`. Sort alphabetically by type.

select account_id,type, sum(amount) as total_amount from bank.trans where account_id=396 group by type order by type asc;

### Query 19
#From the previous output, translate the values for `type` to English, rename the column to `transaction_type`, round `total_amount` down to an integer



select account_id, case type 
when "PRIJEM" THEN 'Incoming' when "VYDAJ" THEN 'Outcoming' else type end as transaction_type,
floor(sum(amount)) as total_amount from bank.trans  
where account_id=396 group by transaction_type order by transaction_type asc;

### Query 20
#From the previous result, modify your query so that it returns only one row, with a column for incoming amount, outgoing amount and the difference.



