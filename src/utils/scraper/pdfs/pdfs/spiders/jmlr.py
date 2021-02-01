import scrapy
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import CrawlSpider, Rule
from ..items import PdfsItem


class JmlrSpider(CrawlSpider):
    name = 'jmlr'
    allowed_domains = ['jmlr.csail.mit.edu']
    start_urls = ['https://jmlr.csail.mit.edu/']


    rules = (
        Rule(LinkExtractor(allow=r'papers/'), callback='parse_item', follow=True),
    )

    def parse_item(self, response):
       file_url = response.css('a[target ="_blank"]::attr(href)').extract()

       if file_url == []:
           file_url = response.css('a[id ="pdf"]::attr(href)').extract()

       if file_url != []:
           file_url = "https://jmlr.csail.mit.edu/" + file_url[0]
           item = PdfsItem()
           item['file_urls'] = [file_url]
           yield item
       
