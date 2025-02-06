
import { useState } from "react";
import { Link } from "react-router-dom";
import { Menu, X, LogIn, UserPlus, Home, Info, Mail } from "lucide-react";

const Navbar = () => {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <nav className="fixed top-0 w-full bg-blue-600 text-white shadow-md z-50">
      <div className="container mx-auto flex justify-between items-center p-4">
        {/* Logo */}
        <h1 className="text-2xl font-bold">MyBrand</h1>

        {/* Desktop Menu */}
        <ul className="hidden md:flex gap-6 text-lg">
          <li>
            <Link to="/" className="flex items-center gap-2 hover:text-gray-300">
              <Home size={18} /> Home
            </Link>
          </li>
          <li>
            <Link to="/about" className="flex items-center gap-2 hover:text-gray-300">
              <Info size={18} /> About
            </Link>
          </li>
          <li>
            <Link to="/contact" className="flex items-center gap-2 hover:text-gray-300">
              <Mail size={18} /> Contact
            </Link>
          </li>
        </ul>

        {/* Buttons */}
        <div className="hidden md:flex gap-4">
          <Link to="/login" className="flex items-center gap-2 bg-white text-blue-600 px-4 py-2 rounded-lg hover:bg-gray-200">
            <LogIn size={18} /> Login
          </Link>
          <Link to="/signup" className="flex items-center gap-2 bg-yellow-400 text-black px-4 py-2 rounded-lg hover:bg-yellow-500">
            <UserPlus size={18} /> Sign Up
          </Link>
        </div>

        {/* Mobile Menu Button */}
        <button className="md:hidden" onClick={() => setIsOpen(!isOpen)}>
          {isOpen ? <X size={28} /> : <Menu size={28} />}
        </button>
      </div>

      {/* Mobile Menu */}
      {isOpen && (
        <ul className="md:hidden bg-blue-700 text-center py-4 space-y-4">
          <li>
            <Link to="/" className="flex justify-center items-center gap-2 hover:text-gray-300">
              <Home size={18} /> Home
            </Link>
          </li>
          <li>
            <Link to="/about" className="flex justify-center items-center gap-2 hover:text-gray-300">
              <Info size={18} /> About
            </Link>
          </li>
          <li>
            <Link to="/contact" className="flex justify-center items-center gap-2 hover:text-gray-300">
              <Mail size={18} /> Contact
            </Link>
          </li>
          <div className="flex flex-col gap-3 mt-4">
            <Link to="/login" className="flex items-center justify-center gap-2 bg-white text-blue-600 px-4 py-2 rounded-lg hover:bg-gray-200">
              <LogIn size={18} /> Login
            </Link>
            <Link to="/signup" className="flex items-center justify-center gap-2 bg-yellow-400 text-black px-4 py-2 rounded-lg hover:bg-yellow-500">
              <UserPlus size={18} /> Sign Up
            </Link>
          </div>
        </ul>
      )}
    </nav>
  );
};

export default Navbar;

