import Home from './components/Home'
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Navbar from './components/Navbar'
import Login from './components/Login'
import Signup from './components/Signup.jsx'
import About from './components/About'
import Footer from './components/Footer'
import Contact from './components/Contact'
import PostDetail from './components/PostDetail'
import ForgotPassword	 from "./components/ForgotPassword.jsx";
import ResetPassword from "./components/ResetPassword.jsx";


const App = () => {
	return (
		  <Router>
			<Navbar />

			<Routes>
				<Route path="/" element={<Home />} />
				<Route path="/posts/:id" element={<PostDetail />} />
				<Route path="/post/:slug" element={<PostDetail />} />
				<Route path="/about" element={<About />} />
				<Route path="/contact" element={<Contact />} />
				<Route path="/login" element={<Login />} />
				<Route path="/signup" element={<Signup />} />
				<Route path="/forgot-password" element={<ForgotPassword />} />
				<Route path="/reset-password" element={<ResetPassword />} />
			</Routes>
			  <Footer />
		</Router>

	)
}

export default App
