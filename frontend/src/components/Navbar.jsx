
import { useState, useEffect } from "react";
import { Link, useNavigate } from "react-router-dom";
import { Menu, X, LogIn, UserPlus, Home, Info, Mail, LogOut, LayoutDashboard } from "lucide-react";
import axios from "axios";

const Navbar = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [user, setUser] = useState(null);
  const navigate = useNavigate();

  useEffect(() => {
    const storedUser = localStorage.getItem("user");
    if (storedUser) {
      setUser(JSON.parse(storedUser));
    }
  }, []);

  const handleLogout = async () => {
    try {
      await axios.post("http://127.0.0.1:8000/logout/");
      localStorage.removeItem("user");
      setUser(null);
      navigate("/login");
    } catch (error) {
      console.error("Logout failed:", error);
    }
  };

  return (
    <nav className="fixed top-0 w-full bg-blue-600 text-white shadow-md z-50">
      <div className="container mx-auto flex justify-between items-center p-4">
        <h1 className="text-2xl font-bold">MyBrand</h1>
        <ul className="hidden md:flex gap-6 text-lg">
          <li><Link to="/" className="flex items-center gap-2 hover:text-gray-300"><Home size={18} /> Home</Link></li>
          <li><Link to="/about" className="flex items-center gap-2 hover:text-gray-300"><Info size={18} /> About</Link></li>
          <li><Link to="/contact" className="flex items-center gap-2 hover:text-gray-300"><Mail size={18} /> Contact</Link></li>
        </ul>
        <div className="hidden md:flex gap-4 items-center">
          {user ? (
            <>
              {user.is_superuser && (
                <Link to="/dashboard" className="flex items-center gap-2 bg-green-500 text-white px-4 py-2 rounded-lg hover:bg-green-600">
                  <LayoutDashboard size={18} /> Dashboard
                </Link>
              )}
              <button onClick={handleLogout} className="flex items-center gap-2 bg-red-500 text-white px-4 py-2 rounded-lg hover:bg-red-600">
                <LogOut size={18} /> Logout
              </button>
            </>
          ) : (
            <>
              <Link to="/login" className="flex items-center gap-2 bg-white text-blue-600 px-4 py-2 rounded-lg hover:bg-gray-200">
                <LogIn size={18} /> Login
              </Link>
              <Link to="/signup" className="flex items-center gap-2 bg-yellow-400 text-black px-4 py-2 rounded-lg hover:bg-yellow-500">
                <UserPlus size={18} /> Sign Up
              </Link>
            </>
          )}
        </div>
        <button className="md:hidden" onClick={() => setIsOpen(!isOpen)}>
          {isOpen ? <X size={28} /> : <Menu size={28} />}
        </button>
      </div>
      {isOpen && (
        <ul className="md:hidden bg-blue-700 text-center py-4 space-y-4">
          <li><Link to="/" className="flex justify-center items-center gap-2 hover:text-gray-300"><Home size={18} /> Home</Link></li>
          <li><Link to="/about" className="flex justify-center items-center gap-2 hover:text-gray-300"><Info size={18} /> About</Link></li>
          <li><Link to="/contact" className="flex justify-center items-center gap-2 hover:text-gray-300"><Mail size={18} /> Contact</Link></li>
          <div className="flex flex-col gap-3 mt-4">
            {user ? (
              <>
                {user.is_superuser && (
                  <Link to="/dashboard" className="flex items-center justify-center gap-2 bg-green-500 text-white px-4 py-2 rounded-lg hover:bg-green-600">
                    <LayoutDashboard size={18} /> Dashboard
                  </Link>
                )}
                <button onClick={handleLogout} className="flex items-center justify-center gap-2 bg-red-500 text-white px-4 py-2 rounded-lg hover:bg-red-600">
                  <LogOut size={18} /> Logout
                </button>
              </>
            ) : null}
          </div>
        </ul>
      )}
    </nav>
  );
};

export default Navbar;
