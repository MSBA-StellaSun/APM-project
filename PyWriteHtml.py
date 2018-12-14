# write-html-2-mac.py
import webbrowser
from fatsecret import Fatsecret
consumer_key = '8575eae8dc11485090730817b5c67c94'
consumer_secret = 'b17201b39a084ac88fd5836d5a58cd29'

#1679
def show(food_name): 
    fs = Fatsecret(consumer_key, consumer_secret)
    foods = fs.foods_search(food_name)
    
    #print("Food Search Results: {}".format(len(foods)))
    #print("{}\n".format(foods))
    food_id = str(foods[0]["food_id"])
    #print(food_id)
    
    
    f = open('helloworld.html','w')
    
    message1 = """<!DOCTYPE html >
    <html>
    <head>
    		<title>Sample Code</title>
    		<style>
    			body
    			{
    				font-family: Arial;
    				font-size: 12px;
    			}
    			.title{
    				font-size: 200%;
    				font-weight:bold;
    				margin-bottom:20px;
    			}
    			.holder{
    				width:300px;
    				margin:0 auto;
    				padding: 10px;
    			}
    		</style>
    		<script src="http://platform.fatsecret.com/js?key=8575eae8dc11485090730817b5c67c94&amp;auto_template=false&amp;theme=none"></script>
    		<script>
    			function doLoad(){
    				fatsecret.setContainer('container');
    				fatsecret.setCanvas("food.get", {food_id: """
                                         
    message2 = """});
    			}
    		</script>
    	</head>
    	<body onload="doLoad()">
    		<div class="holder">
    			<div class="title"><script>fatsecret.writeHolder("foodtitle");</script></div>
    			<script>fatsecret.writeHolder("nutritionpanel");</script>
    			<div id="container"></div>
    		</div>
    	</body>
    </html>"""
    
    f.write(message1 + food_id + message2)
    f.close()
    
    #Change path to reflect file location
    filename = 'file:///Users/wd4446/Box Sync/Adv_Predictive_Modeling/Image_Recognition/model2.0/demo_test/' + 'helloworld.html'
    webbrowser.open_new_tab(filename)
    