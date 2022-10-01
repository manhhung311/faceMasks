const express = require('express');
const app = express();
app.use('/public', express.static(__dirname +'/public'));
app.use('/modelmask',express.static(__dirname +'/modelmask'));
app.use('/modelface',express.static(__dirname + '/modelface'))
app.use(function(req, res, next) {
    res.header("Access-Control-Allow-Origin", "*");
    res.header("Access-Control-Allow-Headers", "Origin, X-Requested-With, Content-Type, Accept");
    next();
});
app.set('views', __dirname + '/views');
app.set('view engine','ejs');
app.get('/', (req, res)=> {
    res.render('index')
})

var port = process.env.PORT || 3000
app.listen(port, (err)=> {
    if(!err) {
        console.log('connect server on port ' + port);
    }
    else  {
        throw err;
    }
})