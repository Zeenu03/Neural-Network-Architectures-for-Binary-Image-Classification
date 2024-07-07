# First Section: Importing Libraries
import os
import requests
from bs4 import BeautifulSoup

# Second Section: Declare important variables
google_image = "https://www.google.com/search?site=&tbm=isch&source=hp&biw=1873&bih=990&"

user_agent = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.77 Safari/537.36"
}



def main():
    download_images()


# Fourth Section: Build the download function
def download_images():
    data = input('What are you looking for? ')
    n_images = int(input('How many images do you want? '))
    
    saved_folder=os.path.join(os.path.dirname(__file__), f'{data} images')
    if not os.path.exists(saved_folder):
        os.mkdir(saved_folder)
    
    print('searching...')

    search_url = google_image + 'q=' + data
    print(search_url)
    response = requests.get(search_url, headers=user_agent)
    # print("response", response)
    html = response.text

    soup = BeautifulSoup(html, 'html.parser')
    # print(soup)
    results = soup.findAll('img', {'class': 'DS1iW'})
    # print("results", results)
    count = 1
    links = []
    for result in results:
        try:
            link = result['src']
            print(link)
            links.append(link)
            count += 1
            if(count > n_images):
                break

        except KeyError:
            print('KeyError')

    print(f"Downloading {len(links)} images...")

    for i, link in enumerate(links):
        response = requests.get(link)

        image_name =  saved_folder + '/' + data + str(i+1) + '.jpg'

        with open(image_name, 'wb') as fh:
            fh.write(response.content)


# Fifth Section: Run your code
if __name__ == "__main__":
    main()