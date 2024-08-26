# BSD 3-Clause License

# Copyright (c) 2024, Guanren Qiao

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from flask import Flask, send_from_directory
import os
app = Flask(__name__)

MEDIA_PATH = './'
play_viedeo="""
<script type=text/javascript>
   const movie_name = {{ movie_name|tojson }};
</script>


  <video id="video" defaultMuted autoplay playsinline controls>
    <source src="{{ url_for('media_video', filename=movie_name) }}" type="video/{{movie_ext}}">
    Your browser does not support the video tag.
  </video>
"""

@app.route('/')
def video_list():
    video_files = [os.path.join(root, f) for root, dirs, files in os.walk(MEDIA_PATH) for f in files if f.endswith('.webm')]
    # Inline HTML template
    template = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Video List</title>
    </head>
    <body>
        <h1>Video List</h1>
        <ul>
            {% for video_file in video_files %}
                <li><a href="{{ url_for('send_media', path=video_file) }}">{{ video_file }}</a></li>
            {% endfor %}
        </ul>
    </body>
    </html>
    """
    return app.jinja_env.from_string(template).render({'video_files': video_files})


@app.route('/media/<path:path>')
def send_media(path):
    """
    :param path: a path like "posts/<int:post_id>/<filename>"
    """
    print(path)
    return send_from_directory(directory="./", path=path, as_attachment=False,conditional=False)

# @app.route("/playvideourl/<filename>")
# def playvideourl(filename): 
#     template="""
# <script type=text/javascript>
#    const movie_name = {{ movie_name|tojson }};
# </script>


#   <video id="video" defaultMuted autoplay playsinline controls>
#     <source src="{{ url_for('send_media', path=movie_name) }}" type="video/mp4">
#     Your browser does not support the video tag.
#   </video>
# """

#     return app.jinja_env.from_string(template).render({'movie_name': "test.mp4"})

if __name__ == '__main__':
    app.run(debug=True,port=1221)
