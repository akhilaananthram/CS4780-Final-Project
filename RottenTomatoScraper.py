__author__ = 'poshengc'

from bs4 import BeautifulSoup
import urllib.request

URL1 = "http://www.rottentomatoes.com/m/the_expendables_3/"
URL2 = "http://www.rottentomatoes.com/m/the_giver/"
URL3 = "http://www.rottentomatoes.com/m/tyler_perrys_a_madea_christmas_2013/"
URL4 = "http://www.rottentomatoes.com/m/what_if_2014/"
URL5 = "http://www.rottentomatoes.com/m/the_november_man/"
URL6 = "http://www.rottentomatoes.com/m/dawn_of_the_planet_of_the_apes/"
URL7 = "http://www.rottentomatoes.com/m/the_hundred_foot_journey/"
URL8 = "http://www.rottentomatoes.com/m/rhymes_for_young_ghouls/"
URL9 = "http://www.rottentomatoes.com/m/and_so_it_goes/"
URL10 = "http://www.rottentomatoes.com/m/22_jump_street/"
URL11 = "http://www.rottentomatoes.com/m/transformers_age_of_extinction/"
URL12 = "http://www.rottentomatoes.com/m/maleficent_2014/"
URL13 = "http://www.rottentomatoes.com/m/the_congress/"
URL14 = "http://www.rottentomatoes.com/m/mr_peabody_and_sherman/"
URL15 = "http://www.rottentomatoes.com/m/how_to_train_your_dragon_2/"
URL16 = "http://www.rottentomatoes.com/m/the_hundred_foot_journey/"
URL17 = "http://www.rottentomatoes.com/m/planes_fire_and_rescue/"
URL18 = "http://www.rottentomatoes.com/m/happy_christmas/"
URL19 = "http://www.rottentomatoes.com/m/jersey_boys/"
URL20 = "http://www.rottentomatoes.com/m/i_am_ali/"
URL21 = "http://www.rottentomatoes.com/m/tammy_2014/"
URL22 = "http://www.rottentomatoes.com/m/the-hero-of-color-city/"
URL23 = "http://www.rottentomatoes.com/m/lets_be_cops/"
URL24 = "http://www.rottentomatoes.com/m/hercules_the_thracian_wars/"
URL25 = "http://www.rottentomatoes.com/m/step_up_all_in/"
URL26 = "http://www.rottentomatoes.com/m/the_fluffy_movie/"
URL27 = "http://www.rottentomatoes.com/m/sex_tape_2014/"
URL28 = "http://www.rottentomatoes.com/m/x_men_days_of_future_past/"
URL29 = "http://www.rottentomatoes.com/m/whitey_united_states_of_america_v_james_j_bulger/"
URL30 = "http://www.rottentomatoes.com/m/into_the_storm_2014/"
URL31 = "http://www.rottentomatoes.com/m/guardians_of_the_galaxy/"
URL32 = "http://www.rottentomatoes.com/m/dolphin_tale_2/"
URL33 = "http://www.rottentomatoes.com/m/when_the_game_stands_tall/"
URL34 = "http://www.rottentomatoes.com/m/frank_2014/"
URL35 = "http://www.rottentomatoes.com/m/i_origins/"
URL36 = "http://www.rottentomatoes.com/m/dead_snow_2_red_vs_dead/"
URL37 = "http://www.rottentomatoes.com/m/the_strange_color_of_your_bodys_tears/"
URL38 = "http://www.rottentomatoes.com/m/as_aboveso_below/"
URL39 = "http://www.rottentomatoes.com/m/the_congress/"
URL40 = "http://www.rottentomatoes.com/m/the-hero-of-color-city/"
URL41 = "http://www.rottentomatoes.com/m/the_hunger_games_mockingjay_part_1/"
URL42 = "http://www.rottentomatoes.com/m/penguins_of_madagascar/"
URL43 = "http://www.rottentomatoes.com/m/horrible_bosses_2/"
URL44 = "http://www.rottentomatoes.com/m/big_hero_6/"
URL45 = "http://www.rottentomatoes.com/m/interstellar_2014/"
URL46 = "http://www.rottentomatoes.com/m/dumb_and_dumber_to/"
URL47 = "http://www.rottentomatoes.com/m/the_theory_of_everything_2014/"
URL48 = "http://www.rottentomatoes.com/m/gone_girl/"
URL49 = "http://www.rottentomatoes.com/m/the_pyramid_2014/"
URL50 = "http://www.rottentomatoes.com/m/st_vincent/"
URL51 = "http://www.rottentomatoes.com/m/hercules_the_thracian_wars/"
URL52 = "http://www.rottentomatoes.com/m/planes_fire_and_rescue/"
URL53 = "http://www.rottentomatoes.com/m/the_christmas_candle_2013/"
URL54 = "http://www.rottentomatoes.com/m/frontera_2014/"
URL55 = "http://www.rottentomatoes.com/m/the_dog_2014/"
URL56 = "http://www.rottentomatoes.com/m/america_imagine_the_world_without_her_2014/"
URL57 = "http://www.rottentomatoes.com/m/life_of_crime_2013/"
URL58 = "http://www.rottentomatoes.com/m/child_of_god_2013/"
URL59 = "http://www.rottentomatoes.com/m/earth_to_echo/"
URL60 = "http://www.rottentomatoes.com/m/a_coffee_in_berlin/"

URL = [URL1, URL2, URL3, URL4, URL5, URL6, URL7, URL8, URL9, URL10,
       URL11, URL12, URL13, URL14, URL15, URL16, URL17, URL18, URL19, URL20,
       URL21, URL22, URL23, URL24, URL25, URL26, URL27, URL28, URL29, URL30,
       URL31, URL32, URL33, URL34, URL35, URL36, URL37, URL38, URL39, URL40,
       URL41, URL42, URL43, URL44, URL45, URL46, URL47, URL48, URL49, URL50,
       URL51, URL52, URL53, URL54, URL55, URL56, URL57, URL58, URL59, URL60]

def get_title(url):
    html = urllib.request.urlopen(url).read()
    title_soup = BeautifulSoup(html)
    title = title_soup.find("span", itemprop="name").get_text()
    title = title.replace("\n", "")
    title = ' '.join(title.split())
    return title

def get_box(url):
    html = urllib.request.urlopen(url).read()
    box_soup = BeautifulSoup(html)
    box_office = box_soup.find("dt", class_="subtle", text="Box Office:").findNextSibling("dd").contents
    return box_office

def get_reviews(url):
    reviews = []
    modified_reviews = []
    html = urllib.request.urlopen(url).read()
    review_soup = BeautifulSoup(html)
    review_sections = review_soup.find("div", {"id": "contentReviews"})
    review_quotes = review_sections.find_all("div", class_="media-body")
    for rev in review_quotes:
        if rev.p is not None:
            reviews.append(rev.p.string)
    for rev in reviews:
        rev = rev.replace("\n", "")
        rev = ' '.join(rev.split())
        modified_reviews.append(rev)
    return modified_reviews

with open("reviews.txt","w") as newfile:
    newfile.write("")
with open("movies.txt","w") as newfile:
    newfile.write("")

review = open("reviews.txt", 'ab')
movie = open("movies.txt", 'ab')

for num in range(0, 60):
    for reviews_f in get_reviews(URL[num]):
        review.write((str(num+1) + "\t" + reviews_f + "\n").encode('utf8'))
    movie.write((str(num+1) + "\t" + get_title(URL[num]) + "\t" + get_box(URL[num])[0] + "\n").encode('utf8'))
    print(str(num+1) + "done")


review.close()
movie.close()