use sakila;

#1.List all films whose length is longer than the average of all the films.
select title from film 
where length > (select avg(length) from film) order by title asc;

#2.How many copies of the film Hunchback Impossible exist in the inventory system?

select count(inventory_id) from inventory where film_id = (select film_id from film where title like "%Hunchback Impossible%");

#3.Use subqueries to display all actors who appear in the film Alone Trip.
select film_id, actor_id, first_name, last_name from actor
join film_actor using (actor_id)
where film_id = (select film_id from film_text where title like "%Alone Trip%");

#4.Sales have been lagging among young families, and you wish to target all family movies for a promotion. 
#Identify all movies categorized as family films.

select title, film_id from film join film_category using(film_id)
 where film_id in  (select film_id from film_category join category using (category_id)
where category_id = (select category_id from category where name like "%Family%"));

#5.Get name and email from customers from Canada using subqueries. Do the same with joins. 
#Note that to create a join, you will have to identify the correct tables with their primary keys and foreign keys, 
#that will help you get the relevant information.

select first_name, last_name, email from customer where address_id in 
(select address_id from address where city_id IN  
(select city_id from city where country_id like 
(select country_id from country where country = "Canada")));

#Optional
#6Which are films starred by the most prolific actor? Most prolific actor is defined as the actor that has acted in the most number of films.
#First you will have to find the most prolific actor and then use that actor_id to find the different films that he/she starred.
select title from film where film_id in 
(select film_id from film_actor where actor_id like 
(select actor_id from film_actor group by actor_id order by count(actor_id) desc limit 1));

#7 Films rented by most profitable customer. 
#You can use the customer table and payment table to find the most profitable customer ie the customer that has made the largest sum of payments
select rental_id from rental where customer_id in 
(select customer_id, sum(amount) as total from payment group by customer_id order by total desc limit 1);

#8Customers who spent more than the average payments(this refers to the average of all amount spent per each customer).

select first_name,last_name from customer where customer_id like 
(select customer_id, sum(amount) from payment group by customer_id having sum(amount) > avg(amount));


