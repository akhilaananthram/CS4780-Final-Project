import urllib2
from bs4 import BeautifulSoup

URL = []
URL.append("http://www.rottentomatoes.com/m/the_expendables_3/")
URL.append("http://www.rottentomatoes.com/m/the_giver/")
URL.append("http://www.rottentomatoes.com/m/tyler_perrys_a_madea_christmas_2013/")
URL.append("http://www.rottentomatoes.com/m/what_if_2014/")
URL.append("http://www.rottentomatoes.com/m/the_november_man/")
URL.append("http://www.rottentomatoes.com/m/dawn_of_the_planet_of_the_apes/")
URL.append("http://www.rottentomatoes.com/m/the_hundred_foot_journey/")
URL.append("http://www.rottentomatoes.com/m/rhymes_for_young_ghouls/")
URL.append("http://www.rottentomatoes.com/m/and_so_it_goes/")
URL.append("http://www.rottentomatoes.com/m/22_jump_street/")
URL.append("http://www.rottentomatoes.com/m/transformers_age_of_extinction/")
URL.append("http://www.rottentomatoes.com/m/maleficent_2014/")
URL.append("http://www.rottentomatoes.com/m/the_congress/")
URL.append("http://www.rottentomatoes.com/m/mr_peabody_and_sherman/")
URL.append("http://www.rottentomatoes.com/m/how_to_train_your_dragon_2/")
URL.append("http://www.rottentomatoes.com/m/the_hundred_foot_journey/")
URL.append("http://www.rottentomatoes.com/m/planes_fire_and_rescue/")
URL.append("http://www.rottentomatoes.com/m/happy_christmas/")
URL.append("http://www.rottentomatoes.com/m/jersey_boys/")
URL.append("http://www.rottentomatoes.com/m/i_am_ali/")
URL.append("http://www.rottentomatoes.com/m/tammy_2014/")
URL.append("http://www.rottentomatoes.com/m/the-hero-of-color-city/")
URL.append("http://www.rottentomatoes.com/m/lets_be_cops/")
URL.append("http://www.rottentomatoes.com/m/hercules_the_thracian_wars/")
URL.append("http://www.rottentomatoes.com/m/step_up_all_in/")
URL.append("http://www.rottentomatoes.com/m/the_fluffy_movie/")
URL.append("http://www.rottentomatoes.com/m/sex_tape_2014/")
URL.append("http://www.rottentomatoes.com/m/x_men_days_of_future_past/")
URL.append("http://www.rottentomatoes.com/m/whitey_united_states_of_america_v_james_j_bulger/")
URL.append("http://www.rottentomatoes.com/m/into_the_storm_2014/")
URL.append("http://www.rottentomatoes.com/m/guardians_of_the_galaxy/")
URL.append("http://www.rottentomatoes.com/m/dolphin_tale_2/")
URL.append("http://www.rottentomatoes.com/m/when_the_game_stands_tall/")
URL.append("http://www.rottentomatoes.com/m/frank_2014/")
URL.append("http://www.rottentomatoes.com/m/i_origins/")
URL.append("http://www.rottentomatoes.com/m/dead_snow_2_red_vs_dead/")
URL.append("http://www.rottentomatoes.com/m/the_strange_color_of_your_bodys_tears/")
URL.append("http://www.rottentomatoes.com/m/as_aboveso_below/")
URL.append("http://www.rottentomatoes.com/m/the_congress/")
URL.append("http://www.rottentomatoes.com/m/the-hero-of-color-city/")
URL.append("http://www.rottentomatoes.com/m/the_hunger_games_mockingjay_part_1/")
URL.append("http://www.rottentomatoes.com/m/penguins_of_madagascar/")
URL.append("http://www.rottentomatoes.com/m/horrible_bosses_2/")
URL.append("http://www.rottentomatoes.com/m/big_hero_6/")
URL.append("http://www.rottentomatoes.com/m/interstellar_2014/")
URL.append("http://www.rottentomatoes.com/m/dumb_and_dumber_to/")
URL.append("http://www.rottentomatoes.com/m/the_theory_of_everything_2014/")
URL.append("http://www.rottentomatoes.com/m/gone_girl/")
URL.append("http://www.rottentomatoes.com/m/the_pyramid_2014/")
URL.append("http://www.rottentomatoes.com/m/st_vincent/")
URL.append("http://www.rottentomatoes.com/m/hercules_the_thracian_wars/")
URL.append("http://www.rottentomatoes.com/m/planes_fire_and_rescue/")
URL.append("http://www.rottentomatoes.com/m/the_christmas_candle_2013/")
URL.append("http://www.rottentomatoes.com/m/frontera_2014/")
URL.append("http://www.rottentomatoes.com/m/the_dog_2014/")
URL.append("http://www.rottentomatoes.com/m/america_imagine_the_world_without_her_2014/")
URL.append("http://www.rottentomatoes.com/m/life_of_crime_2013/")
URL.append("http://www.rottentomatoes.com/m/child_of_god_2013/")
URL.append("http://www.rottentomatoes.com/m/earth_to_echo/")
URL.append("http://www.rottentomatoes.com/m/a_coffee_in_berlin/")

def get_stats(url):
    response = urllib2.urlopen(url)
    html = response.read()
    soup = BeautifulSoup(html, 'html5lib')

    #title
    s = soup.find("span", itemprop="name")
    if s == None:
        return "", "", []
    title = s.get_text()
    title = title.replace("\n", "")
    title = title.strip()
    title = ' '.join(title.split())

    #box office
    s = soup.find("dt", class_="subtle", text="Box Office:")
    if s == None:
        return "", "", []
    box_office = s.findNextSibling("dd").contents[0]

    #reviews
    reviews = []
    modified_reviews = []
    s = soup.find("div", {"id": "contentReviews"})
    if s == None:
        return "", "", []
    review_sections = s

    review_quotes = review_sections.find_all("div", class_="media-body")
    for rev in review_quotes:
        if rev.p is not None:
            reviews.append(rev.p.string)
    for rev in reviews:
        rev = rev.replace("\n", "")
        rev = ' '.join(rev.split())
        modified_reviews.append(rev)

    return title, box_office, modified_reviews

def getAllLinksOnPage(page):
    response = urllib2.urlopen(page)
    html = response.read()

    links = []
    while len(html) != 0:
        start = html.find('"/m/')

        #no more links
        if start == -1:
            break

        html = html[start + 1:]
        end = html.find('"')

        l = html[:end]
        l = l.strip()
        if l != "/m/" and l.count("/") == 3:
            #still on rotten tomatoes
            links.append("http://www.rottentomatoes.com" + l)

        html = html[end + 1:]

    return links

def crawl(seeds, number_to_collect = 250):
    crawled = set()
    print "Starting Crawling"

    count = 0
    frontier = seeds

    #clear old file
    with file("movies-crawled.txt", "w") as f:
        f.write("")
    with file("reviews-crawled.txt", "w") as f:
        f.write("")

    while count < number_to_collect and frontier != []:
        link = frontier.pop()
        if link in crawled:
            continue

        #link is for a movie
        if '/m/' in link:
            title, box_office, reviews = get_stats(link)
            if reviews != []:
                with file("movies-crawled.txt", "a") as f:
                    f.write(str(count) + "\t" + title + "\t" + box_office + "\n")
                with file("reviews-crawled.txt", "a") as f:
                    for r in reviews:
                        f.write(str(count) + "\t" + r.encode('utf-8') + "\n")

                count += 1

        crawled.add(link)
        if count == number_to_collect:
            return

        newLinks = getAllLinksOnPage(link)
        frontier = newLinks + frontier

    #crawl_recursively(seed, crawled)
    return sorted(crawled)

if __name__ == "__main__":
    home = "http://www.rottentomatoes.com"
    top_box_office = "http://www.rottentomatoes.com/browse/in-theaters/?minTomato=0&maxTomato=100&minPopcorn=0&maxPopcorn=100&genres=1;2;4;5;6;8;9;10;11;13;18;14&sortBy=popularity&certified=false"
    dvds = "http://www.rottentomatoes.com/browse/dvd-all/?minTomato=0&maxTomato=100&minPopcorn=0&maxPopcorn=100&services=amazon;amazon_prime;flixster;hbo_go;itunes;netflix_iw;target;vudu&genres=1;2;4;5;6;8;9;10;11;13;18;14&sortBy=release&certified=false"
    crawl(URL + [home, top_box_office, dvds])
