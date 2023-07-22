import openai
import logging
import config



async def load_content_photo(photo):
    pass

async def make_style_transfer(photo):
    return "Everything is ok!"

# async def generate_text(prompt) -> dict:
#     try:
#         response = await openai.ChatCompletion.acreate(
#             model="gpt-3.5-turbo",
#             messages=[
#                 {"role": "user", "content": prompt}
#             ]
#         )
#         return response['choices'][0]['message']['content'], response['usage']['total_tokens']
#     except Exception as e:
#         logging.error(e)
   
# async def generate_image(prompt, n=1, size="1024x1024") -> list[str]:
#     try:
#         response = await openai.Image.acreate(
#             prompt=prompt,
#             n=n,
#             size=size
#         )
#         urls = []
#         for i in response['data']:
#             urls.append(i['url'])
#     except Exception as e:
#         logging.error(e)
#         return []
#     else:
#         return urls