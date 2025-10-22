
movies = {
    "Tamil Padam": ["Comedy", "thriller", "action"],
    "Aayirathil Oruvan": ["sci-fi", "drama", "fanstasy"],
    "The Roboat": ["action", "thriller", "superhero"],
    "TIk TIK TIK": ["sci-fi", "action", "cyberpunk"],
    "Dude": ["romance", "drama"],
    "Lover": ["romance", "drama"],
}


def recommend(movie_name):
    if movie_name not in movies:
        print("Movie not found in database.")
        return
    
    movie_genres = set(movies[movie_name])
    scores = {}

    for movie, genres in movies.items():
        if movie != movie_name:
           
            common = movie_genres.intersection(genres)
            scores[movie] = len(common)

    # Sort by similarity
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    print(f"\nBecause you liked '{movie_name}', you may also like:")
    for movie, score in sorted_scores:
        if score > 0:
            print(f" - {movie} (similar genres: {score})")

recommend("Dude")
