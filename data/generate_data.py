"""
Movie Recommendation System — Dataset Generator
Generates 4 realistic CSVs: movies, users, ratings, reviews
"""
import pandas as pd
import numpy as np
import random
import os

np.random.seed(42)
random.seed(42)

BASE = os.path.dirname(os.path.abspath(__file__))

# ── INDUSTRY CONFIG ───────────────────────────────────────────────────────────
industries = {
    "South Indian": {
        "languages": ["Tamil","Telugu"],
        "actors": ["Vijay","Allu Arjun","Rajinikanth","Ajith Kumar","Mahesh Babu","Prabhas",
                   "Vikram","Dhanush","Ram Charan","Jr NTR","Suriya","Kamal Haasan"],
        "directors": ["Shankar","Atlee","Trivikram","S.S. Rajamouli","Murugadoss","Sukumar",
                      "Vamshi Paidipally","Lokesh Kanagaraj","Mani Ratnam","Gautham Menon"],
    },
    "Bollywood": {
        "languages": ["Hindi"],
        "actors": ["Shah Rukh Khan","Salman Khan","Ranbir Kapoor","Hrithik Roshan",
                   "Aamir Khan","Ranveer Singh","Akshay Kumar","Varun Dhawan"],
        "directors": ["Karan Johar","Sanjay Leela Bhansali","Rohit Shetty","Farhan Akhtar",
                      "Rajkumar Hirani","Imtiaz Ali","Kabir Khan","Zoya Akhtar"],
    },
    "Hollywood": {
        "languages": ["English"],
        "actors": ["Leonardo DiCaprio","Tom Cruise","Chris Evans","Robert Downey Jr",
                   "Dwayne Johnson","Brad Pitt","Matt Damon","Keanu Reeves"],
        "directors": ["Christopher Nolan","Steven Spielberg","James Cameron","Ridley Scott",
                      "David Fincher","Martin Scorsese","Denis Villeneuve","J.J. Abrams"],
    },
    "Marvel": {
        "languages": ["English"],
        "actors": ["Chris Evans","Robert Downey Jr","Chris Hemsworth","Scarlett Johansson",
                   "Benedict Cumberbatch","Paul Rudd","Tom Holland","Chadwick Boseman"],
        "directors": ["Anthony Russo","Joe Russo","Taika Waititi","Ryan Coogler",
                      "Jon Favreau","James Gunn","Scott Derrickson","Peyton Reed"],
    },
    "DC": {
        "languages": ["English"],
        "actors": ["Ben Affleck","Henry Cavill","Gal Gadot","Ezra Miller",
                   "Joaquin Phoenix","Jason Momoa","Margot Robbie","Zachary Levi"],
        "directors": ["Zack Snyder","Patty Jenkins","James Wan","David F. Sandberg",
                      "Todd Phillips","Cathy Yan","Andy Muschietti","Francis Lawrence"],
    },
}

genre_combos = {
    "Action":  [["Action","Drama"],["Action","Thriller"],["Action","Adventure"],["Action","Crime"]],
    "Romance": [["Romance","Drama"],["Romance","Comedy"],["Romance","Action"]],
    "Comedy":  [["Comedy","Drama"],["Comedy","Romance"],["Comedy","Action"]],
    "Sci-Fi":  [["Sci-Fi","Action"],["Sci-Fi","Thriller"],["Sci-Fi","Adventure"],["Sci-Fi","Fantasy"]],
    "Fantasy": [["Fantasy","Adventure"],["Fantasy","Action"],["Fantasy","Drama"]],
    "Horror":  [["Horror","Thriller"],["Horror","Drama"],["Horror","Crime"]],
    "Crime":   [["Crime","Drama"],["Crime","Thriller"],["Crime","Action"]],
    "Drama":   [["Drama","Romance"],["Drama","Crime"],["Drama","Thriller"]],
}

star_boost = {
    "Rajinikanth":20,"Vijay":18,"Allu Arjun":17,"Prabhas":16,"Shah Rukh Khan":19,
    "Salman Khan":17,"Leonardo DiCaprio":16,"Tom Cruise":15,"Robert Downey Jr":18,
    "Chris Evans":15,"Mahesh Babu":15,"Jr NTR":14,"Ram Charan":14,
}

descriptions = {
    "Action":  ["An explosive action saga where the hero battles impossible odds to save the world.",
                "A high-octane thriller packed with breathtaking stunts and edge-of-seat moments.",
                "A relentless action journey where betrayal meets justice in a war for survival."],
    "Romance": ["A heartwarming love story that transcends time and distance.",
                "Two souls find unexpected love in the most unlikely circumstances.",
                "A passionate romance that tests the limits of sacrifice and devotion."],
    "Sci-Fi":  ["A mind-bending science fiction adventure that questions the nature of reality.",
                "Humanity's last hope lies in a mission beyond the stars.",
                "In a world transformed by technology, one man fights for what remains human."],
    "Crime":   ["A gritty crime drama where law and corruption collide in the shadows.",
                "A sharp detective must outsmart a criminal mastermind before time runs out.",
                "In the city's underworld, loyalty is the only currency that matters."],
    "Fantasy": ["An epic fantasy journey through magical realms and ancient prophecies.",
                "A young hero rises to fulfill a destiny written in the stars.",
                "Magic, myth, and courage collide in a battle for an ancient kingdom."],
    "Drama":   ["A powerful human drama exploring love, loss, and the courage to rebuild.",
                "Real emotions, real struggles — a story that will stay with you forever.",
                "A deeply moving tale of family, sacrifice, and redemption."],
    "Horror":  ["Terror lurks in every shadow in this chilling psychological horror.",
                "When the nightmare becomes real, there is no escaping the darkness.",
                "A spine-chilling horror that blurs the line between the living and dead."],
    "Comedy":  ["A laugh-out-loud comedy that finds joy in life's most chaotic moments.",
                "Friendship, fun, and absolute chaos — a comedy you will not forget.",
                "Life is too short not to laugh — this movie proves exactly that."],
}

# Title pools per industry (unique)
south_titles = [f"{a} {s} {y}" for a in ["Thalapathy","Vettai","Kaithi","Sarpatta","Vikram","Leo","Pushpa","Allu","Aarya","Bheemla",
    "Sarileru","Maharshi","Srimanthudu","Julayi","Duvvada","Bahubali","Kalki","Salaar","Adipurush","Mirchi",
    "Rajini","Kabali","Muthu","Darbar","Thalaivar","Thala","Vivegam","Viswasam","Nerkonda","Valimai",
    "Bigil","Mersal","Sarkar","Roar","Iron","Storm","Fire","Force","Rise","Legacy"]
    for s in ["Returns","Legacy","Storm","Rising","War","Force","2","3","Unleashed","Reborn"]
    for y in [""] ][:200]

bollywood_titles = [f"{p} {s}" for p in ["Dilwale","Raees","Don","Zero","Dabangg","Kick","Tiger","Sultan","Bajrangi",
    "Sanju","Brahmastra","Animal","Rockstar","Yeh Jawaani","Dhoom","War","Bang Bang","Fighter","Kabir",
    "Padmaavat","Devdas","Mughal","Bajirao","Sooryavanshi","Simmba","Golmaal","PK","Dangal","Thugs",
    "Mohabbatein","Dil","Kal Ho","Ae Dil","Jab Tak","Raanjhanaa","Tamasha","Barfi","Udta","Queen"]
    for s in ["2","Returns","Legacy","Rising","Storm","Reborn","War","Unleashed","3","Force"] ][:150]

hollywood_titles = [f"{p} {s}" for p in ["Inception","Interstellar","Tenet","Oppenheimer","Mission","Top Gun","Gladiator",
    "Blade Runner","Alien","Prometheus","Once Upon","Django","The Revenant","Wolf","Martian",
    "Elysium","District","John Wick","Matrix","Constantine","The Departed","Goodfellas","Casino",
    "Shutter Island","Gravity","Avatar","Titanic","Aliens","Pandora","Speed Racer"]
    for s in ["2","Returns","Legacy","Rising","Storm","War","Reborn","Unleashed","3","Force"] ][:150]

marvel_titles = [f"{p} {s}" for p in ["Iron Man","Captain America","Thor","Avengers","Spider-Man","Doctor Strange",
    "Black Panther","Ant-Man","Guardians","Scarlet Witch","Black Widow","The Hulk",
    "Nick Fury","Shang-Chi","Eternals","Blade","Nova","X-Men","Fantastic Four","Moon Knight"]
    for s in ["Returns","Legacy","Rising","Storm","War","Reborn","Unleashed","2","3","Force"] ][:100]

dc_titles = [f"{p} {s}" for p in ["Batman","Superman","Wonder Woman","Justice League","The Flash","Aquaman",
    "Shazam","Black Adam","Joker","Harley Quinn","Suicide Squad","Birds of Prey",
    "Green Lantern","Cyborg","Nightwing","Batgirl","Zatanna","Constantine","Lobo","Deathstroke"]
    for s in ["Returns","Legacy","Rising","Storm","War","Reborn","Unleashed","2","3","Dark"] ][:100]

all_titles = {
    "South Indian": south_titles,
    "Bollywood":    bollywood_titles,
    "Hollywood":    hollywood_titles,
    "Marvel":       marvel_titles,
    "DC":           dc_titles,
}

industry_counts = {"South Indian":150,"Bollywood":100,"Hollywood":100,"Marvel":75,"DC":75}

movies = []
movie_id = 1
used = {ind: set() for ind in industries}

for industry, count in industry_counts.items():
    info   = industries[industry]
    titles = all_titles[industry]
    random.shuffle(titles)
    ti = 0

    for _ in range(count):
        while ti < len(titles) and titles[ti] in used[industry]:
            ti += 1
        title = titles[ti] if ti < len(titles) else f"{industry} Movie {movie_id}"
        if ti < len(titles):
            used[industry].add(titles[ti])
            ti += 1

        lang  = random.choice(info["languages"])
        actor = random.choice(info["actors"])
        direc = random.choice(info["directors"])
        year  = random.randint(2000, 2024)

        base_genre = random.choice(list(genre_combos.keys()))
        genre_list = random.choice(genre_combos[base_genre])
        genres     = "|".join(genre_list)

        boost      = star_boost.get(actor, 5)
        popularity = min(100, max(1, int(np.random.normal(52, 18)) + boost))
        rating_avg = round(min(5.0, max(1.5, 1.8 + (popularity/100)*3.0 + np.random.normal(0,0.3))), 1)

        desc_key = genre_list[0] if genre_list[0] in descriptions else "Drama"
        desc     = random.choice(descriptions[desc_key])

        movies.append({
            "movie_id":        movie_id,
            "title":           title,
            "industry":        industry,
            "language":        lang,
            "genres":          genres,
            "main_actor":      actor,
            "director":        direc,
            "release_year":    year,
            "rating_avg":      rating_avg,
            "popularity_score":popularity,
            "description":     desc,
            "poster_url":      f"https://movieposters.example.com/poster_{movie_id}.jpg",
        })
        movie_id += 1

movies_df = pd.DataFrame(movies[:500])
movies_df.to_csv(os.path.join(BASE, "movies.csv"), index=False)
print(f"movies.csv      : {len(movies_df)} rows")

# ── USERS ─────────────────────────────────────────────────────────────────────
first_names = ["Arjun","Priya","Rahul","Deepika","Karthik","Ananya","Vikram","Kavitha",
               "Suresh","Meera","Rohan","Pooja","Arun","Divya","Rajesh","Sneha",
               "Amit","Sunita","Ravi","Lakshmi","James","Emma","Michael","Sarah",
               "David","Olivia","John","Sophia","William","Ava","Sai","Harini",
               "Venkat","Bhavana","Naveen","Swathi","Manoj","Keerthi","Aditya","Nithya",
               "Nikhil","Krithi","Tarun","Simran","Ganesh","Lavanya","Prasad","Varsha"]
last_names  = ["Kumar","Sharma","Reddy","Patel","Singh","Nair","Iyer","Pillai",
               "Verma","Gupta","Rao","Krishnan","Menon","Kapoor","Joshi","Mishra",
               "Smith","Johnson","Brown","Davis","Wilson","Taylor","Anderson","Thomas"]

genres_list    = ["Action","Romance","Comedy","Sci-Fi","Fantasy","Horror","Crime","Drama"]
languages_list = ["Tamil","Telugu","Hindi","English"]

users = []
for uid in range(1, 501):
    users.append({
        "user_id":            uid,
        "name":               f"{random.choice(first_names)} {random.choice(last_names)}",
        "age":                random.randint(18, 60),
        "gender":             random.choice(["Male","Female","Other"]),
        "favorite_genre":     random.choice(genres_list),
        "preferred_language": random.choice(languages_list),
    })

users_df = pd.DataFrame(users)
users_df.to_csv(os.path.join(BASE, "users.csv"), index=False)
print(f"users.csv       : {len(users_df)} rows")

# ── RATINGS ───────────────────────────────────────────────────────────────────
genre_to_movies = {}
for _, row in movies_df.iterrows():
    for g in row["genres"].split("|"):
        genre_to_movies.setdefault(g, [])
        genre_to_movies[g].append(row["movie_id"])

all_mids = movies_df["movie_id"].tolist()
ratings  = []
seen_pairs = set()

for _, user in users_df.iterrows():
    uid      = user["user_id"]
    fav      = user["favorite_genre"]
    pref_lang = user["preferred_language"]
    n_movies = random.randint(20, 50)

    fav_pool  = genre_to_movies.get(fav, all_mids)
    n_fav     = int(n_movies * 0.6)
    n_rand    = n_movies - n_fav

    chosen = set()
    for mid in random.sample(fav_pool, min(n_fav, len(fav_pool))):
        chosen.add(mid)
    for mid in random.sample(all_mids, min(len(all_mids), n_rand + 30)):
        if len(chosen) >= n_movies:
            break
        chosen.add(mid)

    for mid in list(chosen)[:n_movies]:
        if (uid, mid) in seen_pairs:
            continue
        seen_pairs.add((uid, mid))
        movie_row    = movies_df[movies_df.movie_id == mid].iloc[0]
        movie_genres = movie_row["genres"].split("|")
        base         = movie_row["rating_avg"]
        boost        = 0.6 if fav in movie_genres else -0.4
        lang_boost   = 0.3 if movie_row["language"] == pref_lang else 0.0
        r = base + boost + lang_boost + np.random.normal(0, 0.5)
        r = max(1, min(5, int(round(r))))
        ratings.append({"user_id": uid, "movie_id": mid, "rating": r})

ratings_df = pd.DataFrame(ratings).drop_duplicates(subset=["user_id","movie_id"])
ratings_df.to_csv(os.path.join(BASE, "ratings.csv"), index=False)
print(f"ratings.csv     : {len(ratings_df)} rows")

# ── REVIEWS ───────────────────────────────────────────────────────────────────
pos_temps = [
    "Absolutely loved this movie! {actor} was outstanding in every scene.",
    "One of the best films I have seen in years. Highly recommended to everyone!",
    "Brilliant direction by {director}. A true masterpiece of cinema.",
    "The story was gripping from start to finish. Loved every single minute.",
    "{actor} delivered a career-best performance. An absolute must-watch!",
    "Amazing visual effects and a powerful emotional story. Gave it 10 out of 10.",
    "This movie completely blew my mind. {director} is a true genius.",
    "Great entertainment with a strong emotional core. Highly loved it.",
    "Fantastic movie with brilliant performances from the entire cast.",
    "One of the most enjoyable films of the year. Simply superb work!",
    "{actor} was magnetic on screen. Could not take my eyes off the film.",
    "A cinematic experience like no other. {director} outdid himself completely.",
    "Watched it twice already and loved it both times. Pure gold!",
    "Heart-touching story with incredible acting. A film for the ages.",
    "This is what cinema is all about. {actor} at his absolute best.",
]
neu_temps = [
    "It was okay. Some parts were good but the film felt a bit too long.",
    "Decent movie but not the best I have seen this year. Average overall.",
    "Good acting but the story could have been developed much better.",
    "{actor} was good but the screenplay really needed more work overall.",
    "Not bad but had some noticeable pacing issues in the second half.",
    "Watchable once but nothing particularly extraordinary about it.",
    "Some brilliant scenes but overall just average cinema fare.",
    "Mixed feelings about this one. Has its moments but falls short.",
    "Decent effort from the team but falls short of high expectations.",
    "Okay for a one-time watch. Nothing very memorable about the story.",
    "{director} tried something different but it did not fully work out.",
    "The first half was engaging but the second half lost momentum.",
    "Average entertainer. Will appeal to fans of {actor} mostly.",
    "Some good moments scattered throughout but overall just mediocre.",
    "Not bad but not great either. Just a regular commercial movie.",
]
neg_temps = [
    "Disappointing film. Expected much more from {director} after his previous work.",
    "Boring and completely predictable. Totally wasted my time and money.",
    "{actor} tried hard but the script was absolutely terrible to work with.",
    "Could not sit through the whole thing. The story was very poorly written.",
    "Overrated and underwhelming. Not worth all the hype it received at all.",
    "Terrible pacing and very weak characters. Would not recommend to anyone.",
    "One of the worst movies I have watched this entire year. Skip it.",
    "The plot made absolutely no sense. Very confusing and excessively boring.",
    "Waste of a good cast on a poorly written and badly executed story.",
    "Not worth watching at all. Save your time and skip this completely.",
    "Left the theater midway. That should say everything about this film.",
    "{director} completely lost the plot here. A massive disappointment.",
    "The trailer was far better than the actual movie. Felt cheated.",
    "Poor writing, poor direction, poor editing. Nothing works in this film.",
    "A forgettable film that adds nothing new or interesting to the genre.",
]

sentiment_map = {"positive": pos_temps, "neutral": neu_temps, "negative": neg_temps}

reviews     = []
review_id   = 1
seen_reviews = set()

sample_size = min(5000, len(ratings_df))
sample_r    = ratings_df.sample(sample_size, random_state=42)

for _, row in sample_r.iterrows():
    uid    = row["user_id"]
    mid    = row["movie_id"]
    rating = row["rating"]

    if (uid, mid) in seen_reviews:
        continue
    seen_reviews.add((uid, mid))

    sentiment = "positive" if rating >= 4 else ("neutral" if rating == 3 else "negative")
    movie_row = movies_df[movies_df.movie_id == mid].iloc[0]
    template  = random.choice(sentiment_map[sentiment])
    text      = template.replace("{actor}", movie_row["main_actor"]).replace("{director}", movie_row["director"])

    reviews.append({
        "review_id":   review_id,
        "user_id":     uid,
        "movie_id":    mid,
        "review_text": text,
        "sentiment":   sentiment,
    })
    review_id += 1

reviews_df = pd.DataFrame(reviews)
reviews_df.to_csv(os.path.join(BASE, "reviews.csv"), index=False)
print(f"reviews.csv     : {len(reviews_df)} rows")
print(f"\nDuplicate ratings : {ratings_df.duplicated(['user_id','movie_id']).sum()}")
print(f"Duplicate reviews : {reviews_df.duplicated(['user_id','movie_id']).sum()}")
print("\nAll datasets generated successfully!")
