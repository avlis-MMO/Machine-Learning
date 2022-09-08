from bs4 import BeautifulSoup as bs
import requests
import os

directory_of_python_script = os.path.dirname(os.path.abspath(__file__))
url = 'https://www.careerjet.com/python-jobs.html?p='
def find_jobs():
    
    with open(os.path.join(directory_of_python_script, 'Jobs_Posts.txt'), 'w') as f:

        for i in range(3):
            r = requests.get(url+str(i))

            soup = bs(r.text, 'lxml')
            jobs_list = soup.find_all('article', class_ = 'job clicky')
            
            for job in jobs_list:
                if job.find('span', class_ = 'badge badge-r badge-s badge-icon'):
                    company_name = job.find('p', class_ = 'company').text
                    location_name = job.find('ul', class_ = 'location').text.replace(' ','').replace('\n','')
                    more_info = job.header.h2.a['href']
                elif '1' in job.find('span', class_ = 'badge badge-r badge-s').text.replace('\n',''):
                    company_name = job.find('p', class_ = 'company').text
                    location_name = job.find('ul', class_ = 'location').text.replace(' ','').replace('\n','')
                    more_info = job.header.h2.a['href']
                else:
                    continue
            
                f.write(f"Company: {company_name}\n")
                f.write(f"Location: {location_name}\n")
                f.write(f'More info: https://www.careerjet.com{more_info}\n\n')

if __name__ == '__main__':
        find_jobs()


     

