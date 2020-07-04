importScripts("https://www.gstatic.com/firebasejs/7.15.5/firebase-app.js")
importScripts("https://www.gstatic.com/firebasejs/7.15.5/firebase-messaging.js")


// Your web app's Firebase configuration
var firebaseConfig = {
    apiKey: "AIzaSyD6U2mC234IYhqRzwbaZWY_da_GMXLeiUc",
    authDomain: "django-fcm-test-5b82c.firebaseapp.com",
    databaseURL: "https://django-fcm-test-5b82c.firebaseio.com",
    projectId: "django-fcm-test-5b82c",
    storageBucket: "django-fcm-test-5b82c.appspot.com",
    messagingSenderId: "405299081661",
    appId: "1:405299081661:web:70371989391aa8249de2d4",
    measurementId: "G-8PQD6CE540"
};
// Initialize Firebase
firebase.initializeApp(firebaseConfig);

const messaging = firebase.messaging();