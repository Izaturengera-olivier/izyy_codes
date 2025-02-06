
import { useState } from "react";
import axios from "axios";
import { Link } from "react-router-dom";

const Signup = () => {
  const [formData, setFormData] = useState({
    username: "",
    email: "",
    password: "",
    ConfirmPassword: "",
    profilePicture: null, // Added profile picture state
  });
  const [message, setMessage] = useState("");

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleFileChange = (e) => {
    setFormData({ ...formData, profilePicture: e.target.files[0] });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    const formDataToSend = new FormData();
    formDataToSend.append("username", formData.username);
    formDataToSend.append("email", formData.email);
    formDataToSend.append("password", formData.password);
    formDataToSend.append("ConfirmPassword", formData.ConfirmPassword);
    if (formData.profilePicture) {
      formDataToSend.append("profile_picture", formData.profilePicture);
    }

    try {
      const res = await axios.post("http://127.0.0.1:8000/api/register/", formDataToSend, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setMessage(res.data.message);
    } catch (error) {
      setMessage(error.response?.data?.error || "Something went wrong");
    }
  };

  return (
    <div className="flex items-center justify-center min-h-screen bg-gray-100">
      <div className="bg-white p-6 rounded shadow-lg w-96">
        <h2 className="text-2xl font-bold mb-4">Sign Up</h2>
        {message && <p className="text-red-500">{message}</p>}
        <form onSubmit={handleSubmit} className="space-y-3">
          <input type="text" name="username" placeholder="Username" className="w-full p-2 border rounded" onChange={handleChange} required />
          <input type="email" name="email" placeholder="Email" className="w-full p-2 border rounded" onChange={handleChange} required />
          <input type="password" name="password" placeholder="Password" className="w-full p-2 border rounded" onChange={handleChange} required />
          <input type="password" name="ConfirmPassword" placeholder="Confirm Password" className="w-full p-2 border rounded" onChange={handleChange} required />
          {/* Profile Picture Upload */}
          <input type="file" accept="image/*" className="w-full p-2 border rounded" onChange={handleFileChange} />
          <button type="submit" className="w-full bg-blue-500 text-white p-2 rounded">Sign Up</button>
        </form>
        {/* Login Link */}
        <p className="text-center mt-4 text-gray-600">
          Already have an account? <Link to="/login" className="text-blue-500 hover:underline">Login</Link>
        </p>
      </div>
    </div>
  );
};

export default Signup;

