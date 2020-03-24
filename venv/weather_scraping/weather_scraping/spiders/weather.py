import scrapy
import pandas as pd

class SunshineSpider(scrapy.Spider):
    name = 'weather'

    start_urls = [
        'https://en.wikipedia.org/wiki/List_of_cities_by_sunshine_duration'
    ]

    def parse(self, response):
        countries = response.css('tr td:first-child a::text').getall()
        hours = response.css('tr td:nth-child(15)::text').getall()

        pd.DataFrame(countries).to_csv("country_names.csv")
        pd.DataFrame(hours).to_csv("hours.csv")



