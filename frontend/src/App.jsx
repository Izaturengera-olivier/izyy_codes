import Home from './components/Home'
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Navbar from './components/Navbar'
import Login from './components/Login'
import Signup from './components/Signup.jsx'
import About from './components/About'
import Footer from './components/Footer'
import Contact from './components/Contact'


const App = () => {
	return (
		  <Router>
			<Navbar />

			<Routes>
				<Route path="/" element={<Home />} />
				<Route path="/about" element={<About />} />
				<Route path="/contact" element={<Contact />} />
				<Route path="/login" element={<Login />} />
				<Route path="/signup" element={<Signup />} />
			</Routes>
			  <Footer />
		</Router>

	)
}

export default App
