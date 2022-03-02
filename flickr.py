import flickrapi
import webbrowser
import json

f = open("credentials.json")
creds = json.load(f)
API_KEY = creds["API_KEY"]
API_SECRET = creds["API_SECRET"]
USER_OF_INTEREST = creds["USER_OF_INTEREST"]


flickr = flickrapi.FlickrAPI(API_KEY, API_SECRET, format='parsed-json')


# Only do this if we don't have a valid token already
if not flickr.token_valid(perms='read'):

    # Get a request token
    flickr.get_request_token(oauth_callback='oob')

    # Open a browser at the authentication URL
    authorize_url = flickr.auth_url(perms='read')
    webbrowser.open_new_tab(authorize_url)

    # Get the verifier code from the user
    verifier = str(input('Verifier code: '))

    # Trade the request token for an access token
    flickr.get_access_token(verifier)


json_obj = flickr.activity.userComments(
    user=USER_OF_INTEREST, per_page='10', type="comment")


for item in json_obj["items"]["item"]:
    print(item.get("type"), item.get("id"))

    for event in item.get("activity").get("event"):

        print("*", event.get("type"), "by", event.get(
            "username"), ":", event.get("_content"))
        print("\n")

    print("---\n")
